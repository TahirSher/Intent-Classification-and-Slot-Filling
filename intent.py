import os, re, sys, json, abc, warnings, logging, dataclasses, time, math, inspect
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import trange
import gc
import random

import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================
# W&B WRAPPER
# ============================================================

class _WandbDummy:
    def log(self, *a, **kw): pass
    def finish(self): pass
    def watch(self, *a, **kw): pass
    def Histogram(self, *a, **kw): return None
    def Table(self, *a, **kw): return None

_wandb_active = False
wb = _WandbDummy()


def init_wandb(args):
    global wb, _wandb_active
    if not getattr(args, "use_wandb", False):
        logger.info("W&B disabled (pass --use_wandb to enable).")
        return
    try:
        import wandb as _wb
    except ImportError:
        logger.error("wandb not installed. Run: pip install wandb")
        return
    run_name = getattr(args, "wandb_run_name", None) or (
        f"{os.path.basename(args.model_name_or_path)}"
        f"_ee{args.ee_patience}_tau{args.tau_intent}"
        f"_baux{args.bislu_aux_loss_coef}"
        f"_freqexit{int(getattr(args,'use_freq_exit',False))}"
    )
    _wb.init(
        project=getattr(args, "wandb_project", "bislu-pabee"),
        entity=getattr(args, "wandb_entity", None) or None,
        name=run_name,
        config=vars(args),
        settings=_wb.Settings(_disable_stats=False, disable_code=False),
        reinit=True,
    )
    wb = _wb
    _wandb_active = True
    logger.info("W&B run initialised: %s", run_name)


def _gpu_mem_stats(device: str) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    dev = torch.cuda.current_device()
    return {
        "system/gpu_mem_alloc_MB":    torch.cuda.memory_allocated(dev) / 1024 ** 2,
        "system/gpu_mem_reserved_MB": torch.cuda.memory_reserved(dev)  / 1024 ** 2,
        "system/gpu_max_alloc_MB":    torch.cuda.max_memory_allocated(dev) / 1024 ** 2,
    }


# ============================================================
# 1.  INSTRUCTION-FORMAT PARSERS
# ============================================================

def parse_utterance(prompt: str) -> List[str]:
    m = re.search(r'sentence:\s*(.+?)(?:\n|$)', prompt, re.IGNORECASE)
    text = m.group(1).strip() if m else prompt.strip()
    return text.split()


def parse_intents(completion: str) -> str:
    m = re.search(r'intents?:\s*(.+?)(?:\n|$)', completion, re.IGNORECASE)
    if not m:
        return "UNK"
    raw = m.group(1).strip()
    if ',' in raw and '#' not in raw:
        return '#'.join(p.strip() for p in raw.split(',') if p.strip())
    return raw


def parse_slot_pairs(completion: str) -> List[Tuple[str, str]]:
    m = re.search(r'slot_labels?:\s*(.*)', completion, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    pairs = re.findall(r'\[([^\]:]+):\s*([^\]]*)\]', m.group(1))
    return [(tag.strip(), word.strip()) for tag, word in pairs if tag.strip()]


def _greedy_match(utterance_words, span_words, start_from=0):
    n = len(span_words)
    if n == 0 or start_from + n > len(utterance_words):
        return None
    for i in range(start_from, len(utterance_words) - n + 1):
        if utterance_words[i: i + n] == span_words:
            return (i, i + n - 1)
    uw = [w.lower() for w in utterance_words]
    sw = [w.lower() for w in span_words]
    for i in range(start_from, len(uw) - n + 1):
        if uw[i: i + n] == sw:
            return (i, i + n - 1)
    if start_from > 0:
        return _greedy_match(utterance_words, span_words, start_from=0)
    return None


def slot_pairs_to_entities(utterance_words, slot_pairs):
    if not slot_pairs:
        return []
    if len(slot_pairs) == len(utterance_words):
        bio_tags = [tag for tag, _ in slot_pairs]
        raw      = get_bio_entities(bio_tags)
        return [(etype, s, e) for etype, s, e in raw]
    entities    = []
    cur_type    = None
    cur_words   = []
    search_from = 0
    for tag, word in slot_pairs:
        if tag == 'O' or not tag:
            if cur_type and cur_words:
                pos = _greedy_match(utterance_words, cur_words, search_from)
                if pos:
                    entities.append((cur_type, pos[0], pos[1]))
                    search_from = pos[1] + 1
            cur_type, cur_words = None, []
        elif tag.startswith('B-') or tag.upper() == 'B':
            if cur_type and cur_words:
                pos = _greedy_match(utterance_words, cur_words, search_from)
                if pos:
                    entities.append((cur_type, pos[0], pos[1]))
                    search_from = pos[1] + 1
            cur_type  = tag[2:] if tag.startswith('B-') else "_"
            cur_words = [word]
        elif tag.startswith('I-') or tag.upper() == 'I':
            if cur_type is not None:
                cur_words.append(word)
            else:
                cur_type  = tag[2:] if tag.startswith('I-') else "_"
                cur_words = [word]
    if cur_type and cur_words:
        pos = _greedy_match(utterance_words, cur_words, search_from)
        if pos:
            entities.append((cur_type, pos[0], pos[1]))
    return entities


# ============================================================
# 2.  HuggingFace DATASET LOADING
# ============================================================

def load_hf_dataset(
    dataset_name, cache_dir=None,
    dev_split_name="validation", test_split_name="test",
    train_split_name="train", dev_fraction=0.1, test_fraction=0.1,
):
    from datasets import load_dataset
    logger.info("Loading '%s' from HuggingFace Hub ...", dataset_name)
    ds = load_dataset(dataset_name, cache_dir=cache_dir)
    ref_split = train_split_name if train_split_name in ds else list(ds.keys())[0]
    columns   = list(ds[ref_split].features.keys())
    sample    = dict(ds[ref_split][0])
    logger.info("=" * 60)
    logger.info("Dataset columns  : %s", columns)
    logger.info("Available splits : %s", list(ds.keys()))
    for k, v in sample.items():
        logger.info("  %-20s = %r", k, str(v)[:120])
    logger.info("=" * 60)
    if "prompt" in columns and "completion" in columns:
        is_instruction = True
        utt_field  = "prompt"
        int_field  = "completion"
        slot_field = "completion"
        logger.info("Format: instruction-tuning (prompt/completion).")
    else:
        is_instruction = False
        _U = ["text","utterance","sentence","input","seq_in","tokens","words"]
        _I = ["intents","intent_label","intent","label","labels"]
        _S = ["slots","slot_label","seq_out","slot_tags","ner_tags","bio_tags"]
        def _pick(aliases):
            for a in aliases:
                if a in columns: return a
            return None
        utt_field  = _pick(_U)
        int_field  = _pick(_I)
        slot_field = _pick(_S)
        if utt_field is None or int_field is None:
            raise ValueError(f"Cannot detect utterance/intent fields in columns {columns}.")
        logger.info("Format: structured  utterance=%s  intent=%s  slot=%s",
                    utt_field, int_field, slot_field)
    hf_train = ds.get(train_split_name)
    hf_test  = ds.get(test_split_name)
    hf_dev   = ds.get(dev_split_name)
    if hf_train is None:
        raise ValueError(f"No '{train_split_name}' split. Available: {list(ds.keys())}")
    if hf_test is None:
        logger.info("No test split. Carving %.1f%% of training as TEST.", test_fraction * 100)
        tmp      = hf_train.train_test_split(test_size=test_fraction, seed=42)
        hf_train = tmp["train"]; hf_test = tmp["test"]
        logger.info("After test carve: train=%d, test=%d", len(hf_train), len(hf_test))
    if hf_dev is None:
        logger.info("No dev split. Carving %.1f%% of remaining training as DEV.", dev_fraction * 100)
        tmp      = hf_train.train_test_split(test_size=dev_fraction, seed=42)
        hf_train = tmp["train"]; hf_dev = tmp["test"]
        logger.info("After dev carve: train=%d, dev=%d", len(hf_train), len(hf_dev))
    if "id" in hf_train.column_names:
        tr_ids = set(hf_train["id"]); dv_ids = set(hf_dev["id"]); te_ids = set(hf_test["id"])
        overlaps = (tr_ids & dv_ids, tr_ids & te_ids, dv_ids & te_ids)
        labels   = ("Train-Dev", "Train-Test", "Dev-Test")
        if any(overlaps):
            for lbl, ov in zip(labels, overlaps):
                if ov: logger.error("DATA LEAKAGE: %s overlap: %d samples", lbl, len(ov))
            raise RuntimeError("Dataset splits overlap! Check carving logic.")
        else:
            logger.info("No leakage detected between train/dev/test splits.")
    logger.info("=" * 60)
    logger.info("FINAL SPLIT SIZES:  train=%d  dev=%d  test=%d",
                len(hf_train), len(hf_dev), len(hf_test))
    logger.info("=" * 60)
    return hf_train, hf_dev, hf_test, utt_field, int_field, slot_field, is_instruction


def extract_label_sets(hf_train, int_field, slot_field, is_instruction):
    intent_set:    Set[str] = set()
    slot_type_set: Set[str] = set()
    for row in hf_train:
        if is_instruction:
            completion = row["completion"]
            for intent in parse_intents(completion).split('#'):
                intent = intent.strip()
                if intent and intent != "UNK":
                    intent_set.add(intent)
            for tag, _ in parse_slot_pairs(completion):
                if tag.startswith('B-') and len(tag) > 2:
                    slot_type_set.add(tag[2:])
        else:
            raw_int = row[int_field]
            if isinstance(raw_int, list):
                for x in raw_int: intent_set.add(str(x).strip())
            else:
                for x in str(raw_int).replace(',','#').split('#'):
                    if x.strip(): intent_set.add(x.strip())
            if slot_field:
                raw_s = row[slot_field]
                tags  = raw_s if isinstance(raw_s, list) else raw_s.split()
                for tag in tags:
                    tag = str(tag)
                    if tag.startswith('B-') and len(tag) > 2:
                        slot_type_set.add(tag[2:])
    intent_label_set = sorted(intent_set) + ["UNK"]
    slot_label_set   = ["_O_"] + sorted(slot_type_set) + ["UNK"]
    logger.info("Label sets: %d intents, %d slot types.",
                len(intent_label_set), len(slot_label_set))
    return intent_label_set, slot_label_set


# ============================================================
# 2.5  WORD FREQUENCY INDEX
# ============================================================

class WordFrequencyIndex:
    """
    Maps each utterance to a rarity score ∈ [0, 1] based on corpus word
    frequencies.  score=0 → all words frequent → safe to exit early.
    score=1 → rare slot word present → deep layers needed.
    """
    _STOPWORDS: Set[str] = frozenset({
        "a","an","the","is","are","was","were","be","been","being",
        "i","me","my","we","our","you","your","he","she","it","they",
        "do","does","did","to","of","in","on","at","for","with","and",
        "or","but","not","what","which","who","how","when","where","why",
        "can","could","will","would","should","shall","may","might",
        "please","want","need","find","show","tell","get","give","make",
        "like","from","about","between","into","through","during","before",
        "after","above","below","up","down","out","off","over","under",
    })

    def __init__(self, smoothing: float = 0.5, min_freq: int = 1):
        self.smoothing      = smoothing
        self.min_freq       = min_freq
        self._counts:   Dict[str, int]   = {}
        self._log_freq: Dict[str, float] = {}
        self._max_log_freq: float        = 1.0
        self._built = False

    def build(self, hf_split, utterance_field: str, is_instruction: bool) -> None:
        counts: Dict[str, int] = defaultdict(int)
        for row in hf_split:
            if is_instruction:
                words = parse_utterance(row["prompt"])
            else:
                raw   = row[utterance_field]
                words = raw if isinstance(raw, list) else raw.split()
            for w in words:
                counts[w.lower()] += 1
        self._counts = {w: c for w, c in counts.items() if c >= self.min_freq}
        self._log_freq = {w: math.log(c + self.smoothing) for w, c in self._counts.items()}
        self._max_log_freq = max(self._log_freq.values(), default=1.0)
        self._built = True
        logger.info(
            "WordFrequencyIndex built: %d unique words | max_freq=%d | max_log_freq=%.3f",
            len(self._counts), max(self._counts.values(), default=0), self._max_log_freq,
        )

    def word_log_freq_norm(self, word: str) -> float:
        raw = self._log_freq.get(word.lower(), math.log(self.smoothing))
        return max(raw, 0.0) / max(self._max_log_freq, 1e-9)

    def utterance_rarity_score(self, words: List[str]) -> float:
        if not self._built:
            return 0.5
        content = [w for w in words if w.lower() not in self._STOPWORDS]
        if not content:
            content = words
        min_norm_lf = min(self.word_log_freq_norm(w) for w in content)
        return float(np.clip(1.0 - min_norm_lf, 0.0, 1.0))

    def summary_stats(self) -> Dict[str, float]:
        if not self._counts:
            return {}
        counts_arr = np.array(list(self._counts.values()), dtype=np.float64)
        return {
            "freq_index/vocab_size":    len(self._counts),
            "freq_index/mean_freq":     float(counts_arr.mean()),
            "freq_index/median_freq":   float(np.median(counts_arr)),
            "freq_index/max_freq":      float(counts_arr.max()),
            "freq_index/singleton_pct": float((counts_arr == 1).mean() * 100),
        }


# ============================================================
# 3.  BIO PARSING
# ============================================================

def _end_of_chunk(pt,t,pty,ty):
    if pt in ("E","S"): return True
    if pt=="B" and t in ("B","S","O"): return True
    if pt=="I" and t in ("B","S","O"): return True
    if pt not in ("O",".") and pty!=ty: return True
    return False

def _start_of_chunk(pt,t,pty,ty):
    if t in ("B","S"): return True
    if pt in ("E","S","O") and t in ("E","I"): return True
    if t not in ("O",".") and pty!=ty: return True
    return False

def get_bio_entities(seq, suffix=False):
    if any(isinstance(s,list) for s in seq):
        seq = [t for sub in seq for t in sub+["O"]]
    pt,pty,begin = "O","",0; chunks=[]
    for i,chunk in enumerate(seq+["O"]):
        if suffix: t=chunk[-1]; ty=chunk[:-1].rsplit("-",1)[0] or "_"
        else:      t=chunk[0];  ty=chunk[1:].split("-",1)[-1] or "_"
        if _end_of_chunk(pt,t,pty,ty): chunks.append((pty,begin,i-1))
        if _start_of_chunk(pt,t,pty,ty): begin=i
        pt,pty=t,ty
    return chunks


# ============================================================
# 4.  PRECISION / RECALL / F1
# ============================================================

def _prf_divide(num,den,zero_division="warn"):
    mask=den==0.0; den=den.copy(); den[mask]=1; r=num/den
    if not np.any(mask): return r
    r[mask]=0.0 if zero_division in ("warn",0) else 1.0; return r

def _prf(y_true,y_pred,average="micro"):
    et,ep=defaultdict(set),defaultdict(set)
    for i,yt in enumerate(y_true):
        for n,s,e in yt: et[n].add((i,s,e))
    for i,yp in enumerate(y_pred):
        for n,s,e in yp: ep[n].add((i,s,e))
    names=sorted(set(et)|set(ep))
    tp=pred=true=np.array([],dtype=np.int32)
    for n in names:
        a,b=et.get(n,set()),ep.get(n,set())
        tp=np.append(tp,len(a&b)); pred=np.append(pred,len(b)); true=np.append(true,len(a))
    if average=="micro": tp,pred,true=np.array([tp.sum()]),np.array([pred.sum()]),np.array([true.sum()])
    prec=_prf_divide(tp,pred); rec=_prf_divide(tp,true)
    d=prec+rec; d[d==0]=1; f1=2*prec*rec/d
    if average is not None: return np.average(prec),np.average(rec),np.average(f1)
    return prec,rec,f1

def seq_f1(yt,yp):    _,_,f=_prf(yt,yp); return f
def seq_prec(yt,yp):  p,_,_=_prf(yt,yp); return p
def seq_rec(yt,yp):   _,r,_=_prf(yt,yp); return r

def _decode_pred(cate,scores,label_set,flat=True):
    top=[(label_set[cate[i][j].item()],i,j,scores[i][j].item())
         for i in range(len(cate)) for j in range(i,len(cate)) if cate[i][j]>0]
    top.sort(key=lambda x:x[3],reverse=True); res=[]
    for name,ns,ne,_ in top:
        for _,ts,te in res:
            if ns<ts<=ne<te or ts<ns<=te<ne: break
            if flat and (ns<=ts<=te<=ne or ts<=ns<=ne<=te): break
        else: res.append((name,ns,ne))
    return set(res)

def _decode_true(lmat,label_set):
    return [(label_set[lmat[i][j].item()],i,j)
            for i in range(len(lmat)) for j in range(i,len(lmat)) if lmat[i][j]>0]

def get_slot_label_lists(slb,spb,wm,ls):
    yt,yp=[],[]
    for i in range(len(slb)):
        tl=int(wm[i].sum().item()); p2=spb[i][:tl,:tl]; t2=slb[i][:tl,:tl]
        sc,c=p2.max(dim=-1)
        yp.append(list(_decode_pred(c,sc,ls))); yt.append(_decode_true(t2,ls))
    return yt,yp

def compute_metrics(args, ip, il, sp, sl, wm, ls):
    yt, yp = get_slot_label_lists(
        sl.detach().cpu(), sp.detach().float().cpu(), wm.detach().cpu(), ls)
    ip_float = ip.detach().float().cpu()
    il_cpu   = il.detach().cpu()
    single_intent = torch.all(il_cpu.sum(dim=1) == 1).item()
    if single_intent:
        pred_idx = ip_float.argmax(dim=1)
        gold_idx = il_cpu.argmax(dim=1)
        ia = (pred_idx == gold_idx).float().mean().item()
        ipn = torch.zeros_like(il_cpu)
        ipn[torch.arange(il_cpu.size(0)), pred_idx] = 1
        ipn = ipn.numpy(); iln = il_cpu.numpy()
    else:
        probs = torch.sigmoid(ip_float)
        ipn = (probs >= 0.3).numpy()
        iln = il_cpu.numpy()
        ia  = accuracy_score(iln, ipn)
    sfa = float(np.mean(
        np.all(ipn == iln, axis=1) &
        np.array([set(map(tuple,p))==set(map(tuple,t)) for p,t in zip(yp,yt)])
    ))
    f = seq_f1(yt, yp)
    return {
        "intent_acc":       ia,
        "slot_precision":   seq_prec(yt, yp),
        "slot_recall":      seq_rec(yt, yp),
        "slot_f1":          f,
        "mean_intent_slot": (ia + f) / 2.0,
        "semantic_acc":     sfa,
    }


# ============================================================
# 5.  SPAN UTILITIES
# ============================================================

def get_mask(mask):
    return torch.triu(mask.unsqueeze(1).expand(-1, mask.shape[-1], -1))

def get_useful_ones(out, label, mask):
    fm=mask.reshape(-1); fo=out.reshape(-1,out.shape[-1]); fl=label.reshape(-1)
    idx=fm.nonzero(as_tuple=False).squeeze(-1).long()
    return fo.index_select(0,idx), fl.index_select(0,idx)

def get_soft_slot(bs, masks):
    fm=masks.reshape(-1); B,_,_,C=bs.shape
    fs=bs.reshape(-1,C).index_select(0,fm.nonzero(as_tuple=False).squeeze(-1).long())
    soft,start=[],0
    for i in range(B):
        ln=int(masks[i].sum().item()); soft.append(fs[start:start+ln].mean(0,keepdim=True)); start+=ln
    return torch.cat(soft,0).to(bs.device)

def get_useful_embedding(emb, mask):
    B,n1,n2,d=emb.shape; idx=mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1).long()
    return emb.reshape(-1,d).index_select(0,idx)


# ============================================================
# 6.  PYTORCH DATASET
# ============================================================

class HFSLUDataset(Dataset):
    def __init__(self, args, hf_split, utterance_field, intent_field, slot_field,
                 intent_label_set, slot_label_set, tokenizer,
                 is_instruction=True,
                 freq_index: Optional[WordFrequencyIndex] = None):
        self.args            = args
        self.data            = hf_split
        self.utt_field       = utterance_field
        self.int_field       = intent_field
        self.slot_field      = slot_field
        self.tokenizer       = tokenizer
        self.max_seq         = args.max_seq_length + 2
        self.intent_label_id = {w:i for i,w in enumerate(intent_label_set)}
        self.slot_label_id   = {w:i for i,w in enumerate(slot_label_set)}
        self.is_instruction  = is_instruction
        self.freq_index      = freq_index
        self._has_bos        = (tokenizer.bos_token is not None and
                                tokenizer.bos_token != tokenizer.eos_token)
        self._has_eos        = tokenizer.eos_token is not None

    def _tokenise(self, words):
        tokens, wlen = [], []
        if self._has_bos: tokens.append(self.tokenizer.bos_token); wlen.append(1)
        for w in words:
            toks = self.tokenizer.tokenize(w) or [self.tokenizer.unk_token]
            tokens.extend(toks); wlen.append(len(toks))
        if self._has_eos: tokens.append(self.tokenizer.eos_token); wlen.append(1)
        iids  = self.tokenizer.convert_tokens_to_ids(tokens)
        amask = [1]*len(iids); wattn = [1]*len(wlen)
        pad   = self.max_seq - len(wattn)
        if pad > 0: wattn += [0]*pad; wlen += [1]*pad
        return (torch.tensor(iids), torch.tensor(amask),
                torch.tensor(wlen),  torch.tensor(wattn))

    def _span_matrix(self, entities):
        starts, ends, labels = [], [], []
        for etype, es, ee in entities:
            si, ei = es + 1, ee + 1
            if si >= self.max_seq or ei >= self.max_seq: continue
            starts.append(si); ends.append(ei)
            labels.append(self.slot_label_id.get(
                etype, self.slot_label_id.get("UNK", 0)))
        if not starts:
            return torch.zeros(self.max_seq, self.max_seq, dtype=torch.long)
        idx = torch.tensor([starts, ends], dtype=torch.int64)
        val = torch.tensor(labels, dtype=torch.float)
        return torch.sparse.FloatTensor(
            idx, val, torch.Size([self.max_seq, self.max_seq])
        ).to_dense().long()

    def _intent_vec(self, intent_str):
        vec = [0]*len(self.intent_label_id)
        for intent in intent_str.split('#'):
            intent = intent.strip()
            idx = self.intent_label_id.get(intent, self.intent_label_id.get("UNK",0))
            vec[idx] = 1
        return vec

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        if self.is_instruction:
            words      = parse_utterance(row["prompt"])
            intent_str = parse_intents(row["completion"])
            slot_pairs = parse_slot_pairs(row["completion"])
            if len(words) > self.args.max_seq_length:
                words = words[:self.args.max_seq_length]
            entities = slot_pairs_to_entities(words, slot_pairs)
            slot_lbl = self._span_matrix(entities)
        else:
            raw_utt = row[self.utt_field]
            words   = raw_utt if isinstance(raw_utt, list) else raw_utt.split()
            if len(words) > self.args.max_seq_length:
                words = words[:self.args.max_seq_length]
            raw_int    = row[self.int_field]
            intent_str = ('#'.join(str(x) for x in raw_int)
                          if isinstance(raw_int, list)
                          else str(raw_int).replace(',',' ').replace(' ','#'))
            if self.slot_field and row.get(self.slot_field):
                raw_s = row[self.slot_field]
                bio   = raw_s if isinstance(raw_s, list) else raw_s.split()
                ents  = get_bio_entities([str(x) for x in bio])
                slot_lbl = self._span_matrix(ents)
            else:
                slot_lbl = torch.zeros(self.max_seq, self.max_seq, dtype=torch.long)
        iids, amask, wlen, wattn = self._tokenise(words)
        int_lbl = torch.tensor(self._intent_vec(intent_str))
        freq_score = (self.freq_index.utterance_rarity_score(words)
                      if self.freq_index is not None else 0.5)
        return iids, amask, wlen, wattn, int_lbl, slot_lbl, freq_score


def _pad_concat(tensors, pad_value=0):
    ml = max(t.size(0) for t in tensors)
    return torch.stack([F.pad(t.long(),(0,ml-t.size(0)),value=pad_value)
                        if ml>t.size(0) else t.long() for t in tensors])

def collate_fn(batch, pad_id):
    iids, amask, wlen, wattn, ilbl, slbl, fscr = zip(*batch)
    return (
        _pad_concat(iids, pad_id),
        _pad_concat(amask, 0),
        torch.stack(wlen),
        torch.stack(wattn),
        torch.stack(ilbl),
        torch.stack(slbl),
        torch.tensor(fscr, dtype=torch.float),
    )


# ============================================================
# 7.  MISC UTILITIES
# ============================================================

class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience=patience; self.verbose=verbose
        self.counter=0; self.best_score=None; self.early_stop=False
    def __call__(self, val, args):
        s = -val if args.tuning_metric == "loss" else val
        if self.best_score is None: self.best_score=s; self.counter=0
        elif s <= self.best_score:
            self.counter += 1
            if self.verbose: logger.info("EarlyStopping %d/%d", self.counter, self.patience)
            if self.counter >= self.patience: self.early_stop = True
        else: self.best_score=s; self.counter=0

@dataclass
class TrainerState:
    epoch:int=0; global_step:int=0; max_steps:int=0
    num_train_epochs:int=0; loss:float=0.0
    def to_string(self): return json.dumps(dataclasses.asdict(self),sort_keys=True)+"\n"
    def save_to_json(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path,"w") as f:
            f.write(json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True))

def setup_tokenizer(model_name_or_path):
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    if tok.pad_token is None:
        tok.pad_token=tok.eos_token; tok.pad_token_id=tok.eos_token_id
        logger.warning("pad_token set to eos_token ('%s').", tok.eos_token)
    return tok


# ============================================================
# 8.  LOSS FUNCTIONS
# ============================================================

class MLD(nn.Module):
    def forward(self, student, teacher):
        eps=1e-9
        pS=torch.sigmoid(student).clamp(eps,1-eps)
        pT=torch.sigmoid(teacher).clamp(eps,1-eps)
        return (F.kl_div(pS.log(),pT,reduction="sum") +
                F.kl_div((1-pS).log(),(1-pT),reduction="sum")) / (student.numel()+eps)

def intent_loss_func(y_hat, y_true):
    return F.binary_cross_entropy_with_logits(y_hat.float(), y_true.float())

def probe_slot_loss_fn(logits, labels, word_mask):
    B,n,C = logits.shape
    fl,ll,ml = (logits.reshape(B*n,C), labels.reshape(B*n).long(), word_mask.reshape(B*n).bool())
    vl,vll = fl[ml], ll[ml]
    if vll.numel() == 0: return logits.sum() * 0.0
    return F.cross_entropy(vl, vll)

def _stable_supcon_loss(sim, pos):
    eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    sim = sim.float(); pos = pos.bool() & (~eye)
    sim = sim.masked_fill(eye, -1e9)
    log_den  = torch.logsumexp(sim, dim=1, keepdim=True)
    log_prob = sim - log_den
    pos_count = pos.sum(dim=1); valid = pos_count > 0
    if valid.sum() == 0: return sim.sum() * 0.0
    loss_per_row = -(log_prob * pos.float()).sum(dim=1) / pos_count.float().clamp(min=1)
    return loss_per_row[valid].mean()

def scl_intent_loss(embeddings, intent_labels, temp=0.10):
    B, V, d = embeddings.shape; N = B * V
    flat = F.normalize(embeddings.reshape(N, d).float(), dim=-1)
    sim  = flat @ flat.T / temp
    lbl  = intent_labels.unsqueeze(1).expand(B, V, -1).reshape(N, -1).float()
    pos  = (lbl @ lbl.T) > 0
    return _stable_supcon_loss(sim, pos)

def scl_slot_loss(span_emb, slot_labels, temp=0.10):
    N, d = span_emb.shape
    if N <= 1: return span_emb.sum() * 0.0
    norm = F.normalize(span_emb.float(), dim=-1)
    sim  = norm @ norm.T / temp
    pos  = slot_labels.unsqueeze(0) == slot_labels.unsqueeze(1)
    return _stable_supcon_loss(sim, pos)


# ============================================================
# 9.  LAYER MODULES
# ============================================================

class FeedforwardLayer(nn.Module):
    def __init__(self,d,h,dp=0.3):
        super().__init__()
        self.w1=nn.Linear(d,h); self.w2=nn.Linear(h,d)
        self.ln=nn.LayerNorm(d,eps=1e-6); self.dp1=nn.Dropout(dp); self.dp2=nn.Dropout(dp)
    def forward(self,x): r=x; x=self.ln(x); return r+self.dp2(self.w2(self.dp1(F.relu(self.w1(x)))))

class BiaffineLayer(nn.Module):
    def __init__(self,s1,s2,cs):
        super().__init__(); self.cs=cs
        self.bm=nn.Parameter(torch.FloatTensor(s1+1,cs,s2+1))
        nn.init.xavier_uniform_(self.bm.view(s1+1,-1))
    def forward(self,x1,x2):
        B,n,_=x1.shape; o=torch.ones(B,n,1,device=x1.device,dtype=x1.dtype)
        x1=torch.cat((x1,o),-1); x2=torch.cat((x2,o),-1)
        bl=torch.matmul(x1.reshape(-1,x1.shape[-1]),self.bm.reshape(x1.shape[-1],-1))
        bl=bl.reshape(B,n*self.cs,x2.shape[-1])
        return torch.matmul(bl,x2.transpose(1,-1)).reshape(B,n,self.cs,n).transpose(-2,-1)

class IntentClassifier(nn.Module):
    def __init__(self,d,ni,dp=0.0): super().__init__(); self.dp=nn.Dropout(dp); self.lin=nn.Linear(d,ni)
    def forward(self,x): return self.lin(self.dp(x))

class SlotClassifier(nn.Module):
    def __init__(self, cfg, ni, ns, use_ctx=False, dp=0.0, hffw=300, biaffine_dim=128):
        super().__init__()
        self.use_ctx = use_ctx
        h = cfg.hidden_size + ni if use_ctx else cfg.hidden_size
        self.dp  = nn.Dropout(dp)
        self.fs  = FeedforwardLayer(h, hffw)
        self.fe  = FeedforwardLayer(h, hffw)
        self.bi  = BiaffineLayer(h, h, biaffine_dim)
        self.cls = nn.Linear(biaffine_dim, ns)

    def forward(self, wc, ic, wam=None):
        if self.use_ctx:
            ic  = torch.sigmoid(ic).unsqueeze(1).expand(-1, wc.shape[1], -1)
            out = torch.cat((ic, wc), -1)
        else:
            out = wc
        x   = self.dp(out)
        emb = self.bi(self.fs(x), self.fe(x))
        emb = self.dp(emb)
        return self.cls(emb), emb

class EarlyExitIntentHead(nn.Module):
    def __init__(self,d,ni,dp=0.0):
        super().__init__(); self.dp=nn.Dropout(dp); self.lin=nn.Linear(d,ni)
    def forward(self,x): return self.lin(self.dp(x))

class EarlyExitSlotProbe(nn.Module):
    def __init__(self,d,ns,dp=0.0):
        super().__init__(); self.dp=nn.Dropout(dp); self.lin=nn.Linear(d,ns+1)
    def forward(self,x): return self.lin(self.dp(x))


class LastTokenPooling(nn.Module):
    """
    Pool the last non-padding token's hidden state.

    FIX vs. previous version
    ------------------------
    The old code used `h.shape[0]` as the batch size for `torch.arange`.
    Under certain HF / SDPA combinations the hidden tensor `h` can arrive
    with its first dimension equal to T (sequence length) rather than B
    (batch size), causing a shape-mismatch IndexError.

    Using `attn_mask.shape[0]` is always correct: `attn_mask` is the 2-D
    token-level padding mask (B, T) passed from the data loader, and its
    first dimension is unambiguously B.  We additionally clamp `last` to
    prevent out-of-bounds access when the padding mask sums exceed h.shape[1].
    """
    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # attn_mask: (B, T)  — always (batch, seq) from the data loader
        B = attn_mask.shape[0]                          # unambiguous batch size
        last = (attn_mask.sum(dim=1) - 1).clamp(0, h.shape[1] - 1)  # (B,)
        return h[torch.arange(B, device=h.device), last]             # (B, d)


def _build_align(input_ids, words_lengths, device):
    B, ms = input_ids.shape; mw = words_lengths.shape[1]
    align = torch.zeros(B, mw, ms)
    for i, wl in enumerate(words_lengths):
        start = 0
        for j, ln in enumerate(wl):
            ln = int(ln.item())
            if ln > 0: align[i, j, start:start+ln] = 1.0
            start += ln
    return align.to(device)

class DecoderWordRep(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(args.model_name_or_path)
        self.pooling    = LastTokenPooling()

    def forward(self, input_ids, attention_mask, words_lengths):
        out   = self.base_model(input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)
        h_sub = out.last_hidden_state
        align = _build_align(input_ids, words_lengths, h_sub.device).to(dtype=h_sub.dtype)
        return self.pooling(h_sub, attention_mask), torch.bmm(align, h_sub), out.hidden_states


def _build_causal_mask(attention_mask: torch.Tensor,
                       hidden_states: torch.Tensor) -> torch.Tensor:
    """Fallback 4-D additive causal mask (B, 1, T, T)."""
    B, T   = attention_mask.shape
    dtype  = hidden_states.dtype
    device = hidden_states.device
    min_v  = torch.finfo(dtype).min
    causal = torch.triu(torch.full((T, T), min_v, device=device, dtype=dtype), diagonal=1)
    pad    = ((1.0 - attention_mask.float()) * min_v).to(dtype)
    return causal[None, None, :, :] + pad[:, None, None, :]


def _layer_kwargs_for(
    layer: nn.Module,
    causal_mask: Optional[torch.Tensor],
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    position_embeddings: Optional[Tuple],
) -> Dict:
    """
    Build the kwargs dict for a single decoder layer call.

    Uses `inspect.signature` to discover which parameters the layer actually
    accepts, so the code works across transformers versions that renamed
    `past_key_value` → `past_key_values` or added/removed `cache_position`.
    Only keys present in the signature (or any **kwargs catch-all) are passed.
    """
    sig    = inspect.signature(layer.forward)
    params = sig.parameters

    # Check whether the layer accepts arbitrary extra kwargs (**kwargs)
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    candidates: Dict[str, object] = {
        "attention_mask":    causal_mask,
        "position_ids":      position_ids,
        "past_key_value":    None,      # singular (older HF)
        "past_key_values":   None,      # plural   (newer HF)
        "output_attentions": False,
        "use_cache":         False,
        "cache_position":    cache_position,
    }
    if position_embeddings is not None:
        candidates["position_embeddings"] = position_embeddings

    if has_var_kw:
        # Layer accepts **kwargs; pass everything
        return candidates

    # Layer has a fixed signature; only pass what it declares
    return {k: v for k, v in candidates.items() if k in params}


# ============================================================
# 10. JOINT MODEL WITH PABEE + FREQUENCY-ADAPTIVE EXIT
# ============================================================

class JointModelWithEarlyExit(nn.Module):
    def __init__(self, args, num_intent, num_slot):
        super().__init__()
        self.args        = args
        self.num_intent  = num_intent
        self.num_slot    = num_slot
        self.use_freq_exit = getattr(args, "use_freq_exit", False)

        cfg             = AutoConfig.from_pretrained(args.model_name_or_path)
        self.num_layers = cfg.num_hidden_layers
        self.wordrep    = DecoderWordRep(args)

        biaffine_dim     = getattr(args, 'biaffine_dim', 128)
        self.soft_intent = IntentClassifier(cfg.hidden_size, num_intent, args.dropout_rate)
        self.slot_clf    = SlotClassifier(
            cfg, num_intent, num_slot,
            args.use_intent_context_attention,
            args.dropout_rate, args.hidden_dim_ffw, biaffine_dim,
        )
        if args.use_soft_slot:
            self.softmax     = nn.Softmax(dim=-1)
            self.hard_intent = IntentClassifier(
                cfg.hidden_size + num_slot, num_intent, args.dropout_rate
            )

        self.exit_intent_heads = nn.ModuleList([
            EarlyExitIntentHead(cfg.hidden_size, num_intent, args.dropout_rate)
            for _ in range(self.num_layers)])
        self.exit_slot_probes  = nn.ModuleList([
            EarlyExitSlotProbe(cfg.hidden_size, num_slot, args.dropout_rate)
            for _ in range(self.num_layers)])

        _bdt = next(self.wordrep.base_model.parameters()).dtype
        for _cn, _cm in self.named_children():
            if _cn != "wordrep":
                _cm.to(dtype=_bdt)

        if getattr(args, 'use_gc', False):
            if hasattr(self.wordrep.base_model, 'gradient_checkpointing_enable'):
                self.wordrep.base_model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled.")
            else:
                logger.warning("Backbone does not support gradient_checkpointing_enable().")

        raw_mel             = getattr(args, "min_exit_layer", None)
        self.min_exit_layer = raw_mel if raw_mel is not None else self.num_layers // 2
        self.patience       = getattr(args, "ee_patience", 3)
        self.tau_intent     = getattr(args, "tau_intent",  0.05)
        self.tau_slot       = getattr(args, "tau_slot",    0.1)

        logger.info(
            "Model: L=%d  min_exit=%d  patience=%d  tau_intent=%.4f  "
            "biaffine_dim=%d  freq_adaptive_exit=%s",
            self.num_layers, self.min_exit_layer, self.patience,
            self.tau_intent, biaffine_dim, self.use_freq_exit,
        )

    def _bislu_head(self, cls, word_h, wam):
        soft          = self.soft_intent(cls)
        biaffine, seg = self.slot_clf(word_h, soft, wam)
        if self.args.use_soft_slot:
            feat = self.softmax(get_soft_slot(biaffine, wam))
            hard = self.hard_intent(torch.cat([cls, feat], -1))
        else:
            hard = soft
        return cls, seg, soft, hard, biaffine

    def forward(self, input_ids, attention_mask, words_lengths,
                word_attention_mask, return_layer_probes=False):
        device = input_ids.device
        out    = self.wordrep.base_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_layer_probes,
        )
        if return_layer_probes:
            hs     = out.hidden_states
            last_h = hs[-1]
        else:
            last_h = out.last_hidden_state

        align = _build_align(input_ids, words_lengths, device).to(dtype=last_h.dtype)
        cls   = self.wordrep.pooling(last_h, attention_mask)
        main  = self._bislu_head(cls, torch.bmm(align, last_h), word_attention_mask)

        if not return_layer_probes:
            return main

        l_int, l_slot, l_bislu_i, l_bislu_s = [], [], [], []
        for l in range(self.num_layers):
            h      = hs[l + 1]
            cls_l  = self.wordrep.pooling(h, attention_mask)
            wh_l   = torch.bmm(align, h)
            l_int.append(self.exit_intent_heads[l](cls_l))
            l_slot.append(self.exit_slot_probes[l](wh_l))
            _, _, _, hard_l, biaffine_l = self._bislu_head(cls_l, wh_l, word_attention_mask)
            l_bislu_i.append(hard_l)
            l_bislu_s.append(biaffine_l)

        return main, l_int, l_slot, l_bislu_i, l_bislu_s

    def _forward_scl_embeddings(self, inputs):
        with torch.no_grad():
            out = self.wordrep.base_model(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_hidden_states=False,
            )
            h = out.last_hidden_state.detach()
        align  = _build_align(inputs['input_ids'], inputs['words_lengths'], h.device).to(dtype=h.dtype)
        cls    = self.wordrep.pooling(h, inputs['attention_mask'])
        word_h = torch.bmm(align, h)
        return self._bislu_head(cls, word_h, inputs['word_attention_mask'])

    # ------------------------------------------------------------------
    # NOTE: NO @torch.no_grad() here.
    # forward_with_early_exit (the only caller) already carries
    # @torch.no_grad().  Decorating a generator method with a second
    # @torch.no_grad() wraps it in PyTorch's generator_context, which
    # enters and exits the no_grad context at each yield/send boundary.
    # When nested inside another no_grad context this can corrupt the
    # grad-mode state on generator resume, leading to the wrong tensor
    # being received by the caller — which is the root cause of the
    # h.shape[0]=T (instead of B) IndexError seen in practice.
    # ------------------------------------------------------------------
    def _true_layer_iter(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Generator[Tuple[int, torch.Tensor], None, None]:
        """
        Layer-by-layer generator.  Yields (layer_idx, hidden_states) after
        each transformer block.  Breaking the outer for-loop stops computation
        at that layer — remaining blocks are never executed (true FLOPs saving).

        Compatible with transformers ≥ 4.36 (pre-computed RoPE era).
        Uses inspect.signature to pass only the kwargs each layer accepts,
        making the code forward-compatible with API changes across HF versions.
        """
        bm     = self.wordrep.base_model
        device = input_ids.device
        B, T   = input_ids.shape

        if not (hasattr(bm, 'embed_tokens') and hasattr(bm, 'layers')):
            raise RuntimeError(
                "Backbone does not expose .embed_tokens / .layers. "
                "True layer-by-layer early exit is unsupported for this model."
            )

        # ── 1. Token embeddings ──────────────────────────────────────────
        h = bm.embed_tokens(input_ids)                  # (B, T, d)

        # ── 2. Position IDs and cache_position ──────────────────────────
        position_ids   = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        cache_position = torch.arange(T, device=device)

        # ── 3. RoPE embeddings (transformers ≥ 4.43 requires pre-computed) ─
        if hasattr(bm, 'rotary_emb'):
            position_embeddings = bm.rotary_emb(h, position_ids)  # (cos, sin)
        else:
            position_embeddings = None   # older: RoPE computed inside each layer

        # ── 4. 4-D causal attention mask ────────────────────────────────
        causal_mask: Optional[torch.Tensor] = None
        if hasattr(bm, '_update_causal_mask'):
            # Try signatures in order from newest to oldest HF
            for _kw in (
                dict(past_key_values=None, output_attentions=False),
                dict(past_key_values=None),
                {},
            ):
                try:
                    causal_mask = bm._update_causal_mask(
                        attention_mask, h, cache_position, **_kw
                    )
                    break
                except TypeError:
                    continue
        if causal_mask is None and not hasattr(bm, '_update_causal_mask'):
            causal_mask = _build_causal_mask(attention_mask, h)
        # If causal_mask is still None after the loop (SDPA returns None
        # intentionally), that is correct — leave it as None.

        # ── 5. Layer-by-layer iteration ──────────────────────────────────
        for l, layer in enumerate(bm.layers):
            kw = _layer_kwargs_for(
                layer, causal_mask, position_ids, cache_position, position_embeddings
            )
            layer_out = layer(h, **kw)

            # ── Extract hidden states robustly ───────────────────────────
            # Older HF (<=4.42): layer returns a tuple; hidden_states at [0].
            # Newer HF (>=4.43, modeling_layers.py wrapper): layer may return
            # the hidden-state tensor DIRECTLY.  Calling [0] on a 3-D tensor
            # (B, T, d) strips the batch dim and gives (T, d) -- the root
            # cause of the "2-D tensor" RuntimeError seen in practice.
            if isinstance(layer_out, torch.Tensor):
                raw = layer_out                # already (B, T, d)
            elif isinstance(layer_out, (tuple, list)):
                raw = layer_out[0]             # first element is hidden_states
            else:
                raw = layer_out[0]             # ModelOutput or similar

            if raw.dim() != 3:
                raise RuntimeError(
                    f"Layer {l} output has unexpected shape {raw.shape}; "
                    f"expected 3-D (B={B}, T={T}, d)."
                )
            # Some SDPA paths may return (T, B, d) — detect and correct.
            if raw.shape[0] == T and raw.shape[1] == B and T != B:
                logger.warning(
                    "Layer %d output appears transposed (%s); "
                    "correcting to (B, T, d).", l, tuple(raw.shape)
                )
                raw = raw.transpose(0, 1).contiguous()

            h = raw                                      # (B, T, d)
            yield l, h

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_with_early_exit(
        self,
        input_ids:           torch.Tensor,
        attention_mask:      torch.Tensor,
        words_lengths:       torch.Tensor,
        word_attention_mask: torch.Tensor,
        freq_scores: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """
        True layer-by-layer PABEE inference with frequency-adaptive per-sample
        minimum exit layer.

        Computation saving
        ------------------
        `_true_layer_iter` is a plain generator (no @no_grad decorator).
        When `exited.all()` triggers a `break`, the generator frame is
        discarded and the remaining transformer blocks are NEVER executed.
        FLOPs ≈ (max_exit_layer_in_batch + 1) / L of a full forward pass.

        Batch granularity
        -----------------
        Hidden states are computed for the full batch at every layer.
        The `break` fires when the LAST sample in the batch has exited, so
        saved layers = L − max(per_sample_exit_layer).  Per-sample tracking
        is kept for analysis (rarity–exit correlation metric) but does NOT
        give per-sample FLOPs savings within a batch.
        """
        B      = input_ids.size(0)
        device = input_ids.device
        dummy  = next(self.wordrep.base_model.parameters())
        align  = _build_align(input_ids, words_lengths, device).to(dtype=dummy.dtype)

        # ── Per-sample minimum exit layer from frequency rarity ───────────
        if self.use_freq_exit and freq_scores is not None:
            fs      = freq_scores.float().to(device).clamp(0.0, 1.0)
            span    = float(self.num_layers - 1 - self.min_exit_layer)
            per_min = (self.min_exit_layer + fs * span).long().clamp(
                self.min_exit_layer, self.num_layers - 1
            )
        else:
            per_min = torch.full((B,), self.min_exit_layer,
                                 dtype=torch.long, device=device)

        # ── Per-sample PABEE state ────────────────────────────────────────
        pat_cnt  = torch.zeros(B, dtype=torch.long,  device=device)
        exited   = torch.zeros(B, dtype=torch.bool,  device=device)
        exit_lyr = torch.full((B,), self.num_layers - 1,
                              dtype=torch.long, device=device)
        exit_h:  List[Optional[torch.Tensor]] = [None] * B
        prev_ip: Optional[torch.Tensor]        = None
        last_h:  Optional[torch.Tensor]        = None

        for l, h in self._true_layer_iter(input_ids, attention_mask):
            last_h = h                                       # (B, T, d)

            # `attention_mask` is (B, T) from the data loader — always correct.
            cls_l = self.wordrep.pooling(h, attention_mask)  # (B, d)
            ip_l  = torch.sigmoid(self.exit_intent_heads[l](cls_l))  # (B, ni)

            if prev_ip is not None:
                eligible = (~exited) & (l >= per_min)
                delta    = (ip_l - prev_ip).abs().mean(dim=-1)

                stable   = eligible & (delta < self.tau_intent)
                unstable = eligible & (delta >= self.tau_intent)
                pat_cnt  = (pat_cnt + stable.long()) * (~unstable).long()

                new_exits = eligible & (pat_cnt >= self.patience)
                if new_exits.any():
                    for i in new_exits.nonzero(as_tuple=True)[0].tolist():
                        exit_lyr[i] = l
                        exit_h[i]   = h[i].detach().clone()
                    exited = exited | new_exits

            prev_ip = ip_l.detach()

            if exited.all():
                break   # remaining layers are NOT computed

        assert last_h is not None, "No layers were iterated — backbone has no .layers?"
        for i in range(B):
            if exit_h[i] is None:
                exit_h[i] = last_h[i].detach().clone()

        h_exit = torch.stack(exit_h, dim=0)                 # (B, T, d)

        bm = self.wordrep.base_model
        if hasattr(bm, 'norm'):
            h_exit = bm.norm(h_exit)
        elif hasattr(bm, 'ln_f'):
            h_exit = bm.ln_f(h_exit)

        cls_exit = self.wordrep.pooling(h_exit, attention_mask)
        result   = self._bislu_head(
            cls_exit, torch.bmm(align, h_exit), word_attention_mask
        )
        return result, exit_lyr                              # exit_lyr: (B,)


# ============================================================
# 11. TRAINER
# ============================================================

class EarlyExitTrainer:
    def __init__(self, args, tokenizer, train_ds, dev_ds, test_ds,
                 intent_label_set, slot_label_set):
        self.args             = args
        self.device           = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer        = tokenizer
        self.trainer_state    = TrainerState()
        self.intent_label_set = intent_label_set
        self.slot_label_set   = slot_label_set
        self.train_ds         = train_ds
        self.dev_ds           = dev_ds
        self.test_ds          = test_ds
        self.w_drop_out       = [0.3, 0.2]
        self.model = JointModelWithEarlyExit(
            args, len(intent_label_set), len(slot_label_set)
        ).to(self.device)

        if _wandb_active:
            wb.watch(self.model, log="all",
                     log_freq=getattr(args, "wandb_watch_freq", 100), log_graph=False)

    def _dl(self, ds, shuffle):
        s = RandomSampler(ds) if shuffle else SequentialSampler(ds)
        b = self.args.train_batch_size if shuffle else self.args.eval_batch_size
        return DataLoader(
            ds, sampler=s, num_workers=4, batch_size=b,
            collate_fn=lambda x: collate_fn(x, self.tokenizer.pad_token_id),
        )

    def _set_dp(self, model, w):
        for mod in model.wordrep.base_model.modules():
            if isinstance(mod, nn.Dropout): mod.p = w
        return model

    def compute_scl_loss(self, model, cls_o, seg_e, tmp_lbl, int_lbl, masks, inputs):
        if self.args.loss_coef_intent_scl != 0:
            cls_o = cls_o.unsqueeze(1)
        if self.args.loss_coef_slot_scl != 0:
            seg_e = get_useful_embedding(seg_e, masks).unsqueeze(1)
        for p in self.w_drop_out:
            model = self._set_dp(model, p)
            pos_cls, pos_seg, _, _, _ = model._forward_scl_embeddings(inputs)
            if self.args.loss_coef_intent_scl != 0:
                cls_o = torch.cat((cls_o, pos_cls.unsqueeze(1)), 1)
            if self.args.loss_coef_slot_scl != 0:
                seg_e = torch.cat(
                    (seg_e, get_useful_embedding(pos_seg, masks).unsqueeze(1)), 1)
        total = torch.tensor(0.0, device=self.device)
        if self.args.loss_coef_intent_scl != 0:
            total = total + self.args.loss_coef_intent_scl * scl_intent_loss(cls_o, int_lbl)
        if self.args.loss_coef_slot_scl != 0:
            n_views = seg_e.shape[1]
            total = total + self.args.loss_coef_slot_scl * scl_slot_loss(
                seg_e.reshape(-1, seg_e.shape[-1]),
                tmp_lbl.repeat_interleave(n_views),
            )
        return total

    def compute_loss(self, model, inputs, slot_labels, intent_labels, mask,
                     freq_scores: Optional[torch.Tensor] = None):
        """
        Frequency-adaptive depth weighting for auxiliary losses.

        w_l = mean_rarity * (l+1)/L  +  (1 − mean_rarity) * (L−l)/L

        mean_rarity=1 (rare batch)      → deep layers weighted more
        mean_rarity=0 (frequent batch)  → shallow layers weighted more
        mean_rarity=0.5                 → approximately uniform
        """
        main_out, l_int, l_slot, l_bislu_i, l_bislu_s = model(
            **inputs, return_layer_probes=True)
        cls_o, seg_e, soft_i, final_i, biaffine = main_out

        masks = get_mask(mask).to(self.device)
        tmp_out, tmp_lbl = get_useful_ones(biaffine, slot_labels, masks)
        ce    = nn.CrossEntropyLoss(reduction="mean")

        total = (self.args.loss_coef_intent * intent_loss_func(final_i, intent_labels.float())
                 + self.args.loss_coef_slot  * ce(tmp_out, tmp_lbl))

        if self.args.use_sd:
            total = total + self.args.sd_loss_coef * MLD()(soft_i, final_i)
        if self.args.use_scl:
            total = total + self.compute_scl_loss(
                model, cls_o, seg_e, tmp_lbl, intent_labels, masks, inputs)

        L        = len(l_int)
        n_w      = slot_labels.shape[1]
        diag_idx = torch.arange(n_w, device=self.device)
        p_lbl    = slot_labels[:, diag_idx, diag_idx]

        use_freq_loss = (getattr(self.args, "use_freq_exit", False)
                         and freq_scores is not None)
        mean_rarity   = float(freq_scores.mean().item()) if use_freq_loss else 0.5

        aux_probe = torch.tensor(0.0, device=self.device)
        aux_bislu = torch.tensor(0.0, device=self.device)

        for li in range(L):
            w = (mean_rarity * (li + 1) / L
                 + (1.0 - mean_rarity) * (L - li) / L)

            aux_probe = aux_probe + w * (
                self.args.loss_coef_intent * intent_loss_func(l_int[li], intent_labels.float())
                + self.args.loss_coef_slot * probe_slot_loss_fn(l_slot[li], p_lbl, mask).to(self.device)
            )
            b_out, b_lbl = get_useful_ones(l_bislu_s[li], slot_labels, masks)
            aux_bislu = aux_bislu + w * (
                self.args.loss_coef_intent * intent_loss_func(l_bislu_i[li], intent_labels.float())
                + self.args.loss_coef_slot * ce(b_out, b_lbl)
            )

        total = (total
                 + self.args.ee_loss_coef        * aux_probe
                 + self.args.bislu_aux_loss_coef * aux_bislu)
        return total, final_i, biaffine

    def _build_optimizer(self):
        no_decay   = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        trainable  = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        n_train    = sum(p.numel() for _, p in trainable)
        n_total    = sum(p.numel() for p in self.model.parameters())
        logger.info("Optimiser: %d / %d parameters (%.1f%%)",
                    n_train, n_total, 100.0 * n_train / max(n_total, 1))
        return AdamW(
            [
                {"params": [p for n, p in trainable if not any(x in n for x in no_decay)],
                 "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in trainable if     any(x in n for x in no_decay)],
                 "weight_decay": 0.0},
            ],
            lr=self.args.learning_rate, eps=self.args.adam_epsilon,
        )

    def train(self):
        global wb, _wandb_active          # declared first; used throughout method
        dl    = self._dl(self.train_ds, True)
        steps = len(dl) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        opt   = self._build_optimizer()
        sched = get_linear_schedule_with_warmup(
            opt, int(self.args.warmup_proportion * steps), steps)
        use_amp = getattr(self.args, 'use_amp', False) and torch.cuda.is_available()

        logger.info(
            "Training: steps=%d  device=%s  L=%d  min_exit=%d  patience=%d  "
            "tau=%.4f  bislu_aux_coef=%.2f  AMP=%s  freq_adaptive=%s",
            steps, self.device, self.model.num_layers,
            self.model.min_exit_layer, self.model.patience, self.model.tau_intent,
            self.args.bislu_aux_loss_coef, use_amp,
            getattr(self.args, "use_freq_exit", False),
        )
        if _wandb_active:
            wb.log({
                "dataset/train_size":   len(self.train_ds),
                "dataset/dev_size":     len(self.dev_ds),
                "dataset/test_size":    len(self.test_ds),
                "model/num_layers":     self.model.num_layers,
                "model/num_intent":     self.model.num_intent,
                "model/num_slot":       self.model.num_slot,
                "model/min_exit_layer": self.model.min_exit_layer,
            }, step=0)

        es = EarlyStopping(self.args.early_stopping, verbose=True)
        self.model.zero_grad()
        gs = 0
        total_samples_seen = 0
        t_epoch_start      = time.perf_counter()

        for epoch in trange(self.args.num_train_epochs):
            self.model.train()
            ep_loss = 0.0; ep_steps = 0
            t_step  = time.perf_counter()

            for step, batch in enumerate(dl):
                batch_size  = batch[0].size(0)
                freq_scores = batch[6].to(self.device)
                inputs = {
                    "input_ids":           batch[0].to(self.device),
                    "attention_mask":      batch[1].to(self.device),
                    "words_lengths":       batch[2].to(self.device),
                    "word_attention_mask": batch[3].to(self.device),
                }
                opt.zero_grad()
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        loss, _, _ = self.compute_loss(
                            self.model, inputs,
                            batch[5].to(self.device), batch[4].to(self.device),
                            batch[3], freq_scores=freq_scores)
                else:
                    loss, _, _ = self.compute_loss(
                        self.model, inputs,
                        batch[5].to(self.device), batch[4].to(self.device),
                        batch[3], freq_scores=freq_scores)

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                if not torch.isfinite(loss):
                    logger.warning("Non-finite loss epoch=%d step=%d. Skipping.", epoch, step)
                    opt.zero_grad(set_to_none=True); self.model.zero_grad(set_to_none=True)
                    continue

                ep_loss += loss.item(); ep_steps += 1
                loss.backward()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm)
                    if not torch.isfinite(grad_norm):
                        logger.warning("Non-finite grad norm epoch=%d step=%d. Skipping.", epoch, step)
                        opt.zero_grad(set_to_none=True); self.model.zero_grad(set_to_none=True)
                        continue
                    opt.step(); sched.step(); self.model.zero_grad(set_to_none=True)
                    gs += 1; total_samples_seen += batch_size
                    self.trainer_state.epoch       = epoch
                    self.trainer_state.global_step = gs
                    self.trainer_state.max_steps   = steps
                    self.trainer_state.loss        = ep_loss / max(ep_steps, 1)

                    if _wandb_active:
                        t_now   = time.perf_counter()
                        elapsed = max(t_now - t_step, 1e-6)
                        t_step  = t_now
                        wb.log({
                            "train/loss":              loss.item() * self.args.gradient_accumulation_steps,
                            "train/loss_smoothed":     ep_loss / max(ep_steps, 1),
                            "train/learning_rate":     sched.get_last_lr()[0],
                            "train/grad_norm":         grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm),
                            "perf/samples_per_sec":    batch_size / elapsed,
                            "perf/total_samples_seen": total_samples_seen,
                            "train/epoch":             epoch + (step + 1) / len(dl),
                            "train/batch_mean_rarity": freq_scores.mean().item(),
                            **_gpu_mem_stats(self.device),
                        }, step=gs)

                if (step + 1) % self.args.logging_steps == 0:
                    logger.info(self.trainer_state.to_string())

            epoch_loss = ep_loss / max(ep_steps, 1)
            epoch_time = time.perf_counter() - t_epoch_start
            t_epoch_start = time.perf_counter()
            if _wandb_active:
                wb.log({"epoch/train_loss": epoch_loss,
                        "epoch/epoch_time_sec": epoch_time,
                        "epoch/epoch": epoch}, step=gs)

            results = self.evaluate("dev", global_step=gs, epoch=epoch)
            es(results[self.args.tuning_metric], self.args)
            if es.counter == 0: self.save_model()
            if es.early_stop: logger.info("Early stopping."); break

        wb.finish()
        # Reset to dummy so that a subsequent evaluate("test") call
        # never tries to log to a closed run.
        wb = _WandbDummy()
        _wandb_active = False

    def evaluate(self, mode="dev", global_step: int = 0, epoch: int = 0):
        ds = {"dev": self.dev_ds, "test": self.test_ds}.get(mode)
        if ds is None: raise ValueError(f"mode must be dev or test, got {mode!r}")
        logger.info("Eval [%s] %d samples", mode, len(ds))
        dl = self._dl(ds, False)
        self.model.eval()

        ev_loss = 0.0
        int_la, int_pa, slot_la, slot_pa, mask_a = [], [], [], [], []
        all_exit_lyrs:   List[int]   = []
        all_freq_scores: List[float] = []
        layer_exit_counts = defaultdict(int)
        ce = nn.CrossEntropyLoss(reduction="mean")
        t_eval_start = time.perf_counter()

        for batch in dl:
            freq_scores = batch[6].to(self.device)
            with torch.no_grad():
                inputs = {
                    "input_ids":           batch[0].to(self.device),
                    "attention_mask":      batch[1].to(self.device),
                    "words_lengths":       batch[2].to(self.device),
                    "word_attention_mask": batch[3].to(self.device),
                }
                il = batch[4].to(self.device)
                sl = batch[5].to(self.device)

                out, exit_lyr_batch = self.model.forward_with_early_exit(
                    **inputs, freq_scores=freq_scores)
                cls_o, seg_e, soft_i, final_i, biaffine = out

                for li in exit_lyr_batch.tolist():
                    all_exit_lyrs.append(li)
                    layer_exit_counts[li] += 1
                all_freq_scores.extend(freq_scores.cpu().tolist())

                masks = get_mask(batch[3]).to(self.device)
                to, tl = get_useful_ones(biaffine, sl, masks)
                ev_loss += (
                    self.args.loss_coef_intent * intent_loss_func(final_i, il.float())
                    + self.args.loss_coef_slot * ce(to, tl)
                ).item()
            int_la.append(il); int_pa.append(final_i)
            slot_la.append(sl); slot_pa.append(biaffine); mask_a.append(batch[3])

        eval_time = time.perf_counter() - t_eval_start
        ev_loss  /= len(dl)
        results   = {"loss": ev_loss}
        results.update(compute_metrics(
            self.args,
            torch.cat(int_pa, 0), torch.cat(int_la, 0),
            torch.cat(slot_pa, 0), torch.cat(slot_la, 0),
            torch.cat(mask_a, 0), self.slot_label_set,
        ))
        et = torch.tensor(all_exit_lyrs, dtype=torch.float)
        ml = self.model.num_layers - 1
        me = et.mean().item()
        se = et.std().item() if len(et) > 1 else 0.0
        results.update({
            "mean_exit_layer":   me, "std_exit_layer":    se,
            "pct_full_pass":     (et == ml).float().mean().item(),
            "layer_savings_pct": 1.0 - me / max(ml, 1),
        })

        for k in sorted(results):
            logger.info("  %-25s = %s", k, results[k])
        logger.info("  Exit: mean=%.2f std=%.2f full_pass=%.1f%% savings=%.1f%%",
                    me, se, results["pct_full_pass"]*100, results["layer_savings_pct"]*100)
        logger.info("  Exit layer distribution:")
        for li in sorted(layer_exit_counts):
            pct = 100.0 * layer_exit_counts[li] / max(len(all_exit_lyrs), 1)
            logger.info("    layer %2d : %d samples (%.1f%%)", li, layer_exit_counts[li], pct)

        if _wandb_active:
            prefix   = "dev" if mode == "dev" else "test"
            log_dict: Dict = {f"{prefix}/{k}": v for k, v in results.items()}
            log_dict[f"{prefix}/eval_time_sec"]   = eval_time
            log_dict[f"{prefix}/samples_per_sec"] = len(ds) / max(eval_time, 1e-6)

            if all_exit_lyrs:
                log_dict[f"{prefix}/exit_layer_histogram"] = wb.Histogram(
                    all_exit_lyrs, num_bins=self.model.num_layers)
                table = wb.Table(columns=["layer", "sample_count", "pct"])
                for li in range(self.model.num_layers):
                    cnt = layer_exit_counts.get(li, 0)
                    table.add_data(li, cnt, 100.0 * cnt / max(len(all_exit_lyrs), 1))
                log_dict[f"{prefix}/exit_layer_table"] = table

            if all_freq_scores and getattr(self.args, "use_freq_exit", False):
                fs_arr  = np.array(all_freq_scores)
                el_arr  = np.array(all_exit_lyrs, dtype=np.float32)
                q_edges = np.quantile(fs_arr, [0.0, 0.25, 0.50, 0.75, 1.0])
                q_labels = ["Q1_frequent", "Q2", "Q3", "Q4_rare"]
                strat_table = wb.Table(
                    columns=["quartile","n_samples","rarity_mean",
                             "mean_exit_layer","pct_savings"])
                for qi in range(4):
                    mask = (fs_arr >= q_edges[qi]) & (fs_arr <= q_edges[qi + 1])
                    if not mask.any(): continue
                    el_q  = float(el_arr[mask].mean())
                    sav_q = 1.0 - el_q / max(ml, 1)
                    strat_table.add_data(q_labels[qi], int(mask.sum()),
                                         float(fs_arr[mask].mean()), el_q, sav_q)
                    log_dict[f"{prefix}/freq_strat/{q_labels[qi]}/mean_exit_layer"] = el_q
                    log_dict[f"{prefix}/freq_strat/{q_labels[qi]}/layer_savings_pct"] = sav_q
                log_dict[f"{prefix}/freq_stratified_exit_table"] = strat_table
                if len(fs_arr) > 2:
                    corr = float(np.corrcoef(fs_arr, el_arr)[0, 1])
                    log_dict[f"{prefix}/rarity_exit_correlation"] = corr
                    logger.info("  Rarity–exit correlation: %.4f (expected > 0)", corr)

            log_dict.update(_gpu_mem_stats(self.device))
            log_dict["epoch/epoch"] = epoch
            wb.log(log_dict, step=global_step)

        self._write(f"eval_{mode}_results.txt", results)
        return results

    def save_model(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        save_path = os.path.join(self.args.output_dir, "checkpoint.pth")
        torch.save({"state_dict":       self.model.state_dict(),
                    "intent_label_set": self.intent_label_set,
                    "slot_label_set":   self.slot_label_set}, save_path)
        torch.save(self.args, os.path.join(self.args.output_dir, "training_args.bin"))
        self.trainer_state.save_to_json(
            os.path.join(self.args.output_dir, "trainer_state.json"))
        logger.info("Saved -> %s", save_path)

    def load_model(self):
        ckpt_path = os.path.join(self.args.output_dir, "checkpoint.pth")
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["state_dict"], strict=True)
        self.model.to(self.device); self.model.eval()
        logger.info("Loaded -> %s", ckpt_path)

    def _write(self, fname, results):
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(os.path.join(self.args.output_dir, fname), "w") as f:
            [f.write(f"{k} = {v}\n") for k, v in sorted(results.items())]


# ============================================================
# 12. MAIN + ARGPARSE
# ============================================================

def main(args):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.gpu is not None and torch.cuda.is_available():
        if args.gpu < torch.cuda.device_count():
            torch.cuda.set_device(args.gpu)
            _gp = torch.cuda.get_device_properties(args.gpu)
            print(f"GPU {args.gpu}: {_gp.name} ({_gp.total_memory/1024**3:.1f} GB)")
        else:
            raise RuntimeError(f"GPU {args.gpu} not available ({torch.cuda.device_count()} found)")
    else:
        print("CUDA not available, using CPU" if not torch.cuda.is_available()
              else f"GPU 0: {torch.cuda.get_device_properties(0).name}")

    init_wandb(args)

    hf_train, hf_dev, hf_test, utt_f, int_f, slot_f, is_instr = load_hf_dataset(
        dataset_name=args.hf_dataset, cache_dir=args.cache_dir or None,
        dev_split_name=args.dev_split, test_split_name=args.test_split,
        train_split_name=args.train_split,
        dev_fraction=args.dev_fraction, test_fraction=args.test_fraction,
    )
    intent_label_set, slot_label_set = extract_label_sets(hf_train, int_f, slot_f, is_instr)
    tokenizer = setup_tokenizer(args.model_name_or_path)

    freq_index = WordFrequencyIndex(
        smoothing=getattr(args, "freq_smoothing", 0.5),
        min_freq=getattr(args, "freq_min_count", 1),
    )
    freq_index.build(hf_train, utt_f, is_instr)
    if _wandb_active:
        wb.log(freq_index.summary_stats(), step=0)

    make_ds = lambda split_data: HFSLUDataset(
        args=args, hf_split=split_data,
        utterance_field=utt_f, intent_field=int_f, slot_field=slot_f,
        intent_label_set=intent_label_set, slot_label_set=slot_label_set,
        tokenizer=tokenizer, is_instruction=is_instr, freq_index=freq_index,
    )
    trainer = EarlyExitTrainer(
        args=args, tokenizer=tokenizer,
        train_ds=make_ds(hf_train) if args.do_train else None,
        dev_ds=make_ds(hf_dev), test_ds=make_ds(hf_test),
        intent_label_set=intent_label_set, slot_label_set=slot_label_set,
    )
    if args.do_train: trainer.train()
    if args.do_eval:  trainer.load_model(); trainer.evaluate("test")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="BiSLU + PABEE | Frequency-Adaptive Early Exit | HF datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--hf_dataset",    required=True)
    p.add_argument("--cache_dir",     default="")
    p.add_argument("--train_split",   default="train")
    p.add_argument("--dev_split",     default="validation")
    p.add_argument("--test_split",    default="test")
    p.add_argument("--dev_fraction",  default=0.1,  type=float)
    p.add_argument("--test_fraction", default=0.1,  type=float)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--output_dir",         required=True)
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_eval",  action="store_true")
    p.add_argument("--max_seq_length",              default=100,  type=int)
    p.add_argument("--train_batch_size",            default=16,   type=int)
    p.add_argument("--eval_batch_size",             default=8,    type=int)
    p.add_argument("--learning_rate",               default=2e-5, type=float)
    p.add_argument("--num_train_epochs",            default=15,   type=int)
    p.add_argument("--warmup_proportion",           default=0.1,  type=float)
    p.add_argument("--gradient_accumulation_steps", default=2,    type=int)
    p.add_argument("--weight_decay",                default=0.01, type=float)
    p.add_argument("--adam_epsilon",                default=1e-8, type=float)
    p.add_argument("--max_grad_norm",               default=1.0,  type=float)
    p.add_argument("--logging_steps",               default=200,  type=int)
    p.add_argument("--early_stopping",              default=5,    type=int)
    p.add_argument("--tuning_metric",               default="mean_intent_slot")
    p.add_argument("--loss_coef_intent",     default=0.5,  type=float)
    p.add_argument("--loss_coef_slot",       default=0.5,  type=float)
    p.add_argument("--loss_coef_slot_scl",   default=0.5,  type=float)
    p.add_argument("--loss_coef_intent_scl", default=0.5,  type=float)
    p.add_argument("--sd_loss_coef",         default=0.5,  type=float)
    p.add_argument("--use_soft_slot",                action="store_true")
    p.add_argument("--use_scl",                      action="store_true")
    p.add_argument("--use_sd",                       action="store_true")
    p.add_argument("--use_intent_context_attention", action="store_true")
    p.add_argument("--dropout_rate",   default=0.1,  type=float)
    p.add_argument("--hidden_dim_ffw", default=300,  type=int)
    p.add_argument("--min_exit_layer", default=None, type=int)
    p.add_argument("--ee_patience",   default=3,    type=int)
    p.add_argument("--tau_intent",    default=0.05, type=float)
    p.add_argument("--tau_slot",      default=0.1,  type=float)
    p.add_argument("--ee_loss_coef",  default=0.3,  type=float)
    p.add_argument("--bislu_aux_loss_coef", default=0.3, type=float)
    p.add_argument("--use_freq_exit", action="store_true",
                   help="Frequency-adaptive per-sample min exit + depth-weighted aux loss.")
    p.add_argument("--freq_smoothing",  default=0.5, type=float)
    p.add_argument("--freq_min_count",  default=1,   type=int)
    p.add_argument("--biaffine_dim", default=128, type=int)
    p.add_argument("--use_gc",  action="store_true")
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--use_wandb",        action="store_true")
    p.add_argument("--wandb_project",    default="bislu-pabee")
    p.add_argument("--wandb_entity",     default=None)
    p.add_argument("--wandb_run_name",   default=None)
    p.add_argument("--wandb_watch_freq", default=100, type=int)

    args = p.parse_args()
    if not args.do_train and not args.do_eval:
        p.error("Specify --do_train and/or --do_eval.")
    if args.use_scl and not args.use_soft_slot:
        p.error("--use_scl requires --use_soft_slot.")
    if not args.use_soft_slot:
        logger.warning("--use_soft_slot not set; soft-slot feedback disabled.")
    main(args)