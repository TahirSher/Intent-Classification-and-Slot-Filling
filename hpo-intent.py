import os, re, sys, json, abc, warnings, logging, dataclasses, time, math, inspect
import copy, itertools, sqlite3, contextlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
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

def _reconstruct_words(ds: "HFSLUDataset", idx: int) -> List[str]:

    row = ds.data[idx]
    if ds.is_instruction:
        words = parse_utterance(row["prompt"])
    else:
        raw_utt = row[ds.utt_field]
        words = raw_utt if isinstance(raw_utt, list) else raw_utt.split()
    if len(words) > ds.args.max_seq_length:
        words = words[:ds.args.max_seq_length]
    return words


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
        eps = 1e-4  # 1e-9, must exceed bf16/fp32 rounding resolution near 0/1, not just be "small"
        pS = torch.sigmoid(student.float()).clamp(eps, 1 - eps)
        pT = torch.sigmoid(teacher.float()).clamp(eps, 1 - eps)
        return (F.kl_div(pS.log(), pT, reduction="sum") +
                F.kl_div((1 - pS).log(), (1 - pT), reduction="sum")) / (student.numel() + eps)

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

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        
        B = attn_mask.shape[0]                          
        last = (attn_mask.sum(dim=1) - 1).clamp(0, h.shape[1] - 1)  
        return h[torch.arange(B, device=h.device), last]             


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
        load_dtype_name = getattr(args, "load_dtype", "fp32")
        if load_dtype_name == "bf16":

            logger.warning(
                "--load_dtype=bf16: loading backbone weights AND optimizer "
                "state in bfloat16. This trades memory for numerical "
                "precision beyond what --use_amp alone does. Monitor "
                "train/nonfinite_total closely; if you see repeated "
                "non-finite events, switch back to --load_dtype=fp32 "
                "(with --use_amp for the compute-side memory saving instead)."
            )
            self.base_model = AutoModel.from_pretrained(
                args.model_name_or_path, torch_dtype=torch.bfloat16)
        else:
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
        
        return candidates


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

        if getattr(args, 'use_gc', False) and not getattr(args, 'freeze_backbone', False):
            if hasattr(self.wordrep.base_model, 'gradient_checkpointing_enable'):
                self.wordrep.base_model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled.")
            else:
                logger.warning("Backbone does not support gradient_checkpointing_enable().")
        elif getattr(args, 'use_gc', False) and getattr(args, 'freeze_backbone', False):
            logger.info("Gradient checkpointing skipped: backbone is frozen, no backward pass through it is needed.")

        raw_mel             = getattr(args, "min_exit_layer", None)
        self.min_exit_layer = raw_mel if raw_mel is not None else self.num_layers // 2
        self.patience       = getattr(args, "ee_patience", 3)
        self.tau_intent     = getattr(args, "tau_intent",  0.05)
        self.tau_slot       = getattr(args, "tau_slot",    0.1)
        self.freeze_backbone = getattr(args, "freeze_backbone", False)

        if self.freeze_backbone:
            for p in self.wordrep.base_model.parameters():
                p.requires_grad_(False)
            n_frozen = sum(p.numel() for p in self.wordrep.base_model.parameters())
            n_total  = sum(p.numel() for p in self.parameters())
            logger.info(
                "Backbone FROZEN: %d/%d params (%.2f%%) excluded from optimization. "
                "Only BiSLU+PABEE heads (soft_intent, hard_intent, slot_clf, "
                "exit_intent_heads, exit_slot_probes) remain trainable.",
                n_frozen, n_total, 100.0 * n_frozen / max(n_total, 1),
            )

        logger.info(
            "Model: L=%d  min_exit=%d  patience=%d  tau_intent=%.4f  "
            "biaffine_dim=%d  freq_adaptive_exit=%s  freeze_backbone=%s",
            self.num_layers, self.min_exit_layer, self.patience,
            self.tau_intent, biaffine_dim, self.use_freq_exit, self.freeze_backbone,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone and not getattr(self.args, "use_scl", False):
            self.wordrep.base_model.eval()
        return self

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
        backbone_ctx = torch.no_grad() if self.freeze_backbone else contextlib.nullcontext()
        with backbone_ctx:
            out = self.wordrep.base_model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=return_layer_probes,
            )
            if return_layer_probes:
                hs     = out.hidden_states
                last_h = hs[-1]
            else:
                last_h = out.last_hidden_state
        if self.freeze_backbone:
            last_h = last_h.detach()
            if return_layer_probes:
                hs = tuple(h.detach() for h in hs)

        align = _build_align(input_ids, words_lengths, device).to(dtype=last_h.dtype)
        cls   = self.wordrep.pooling(last_h, attention_mask)
        main  = self._bislu_head(cls, torch.bmm(align, last_h), word_attention_mask)

        if not return_layer_probes:
            return main

        use_layer_ckpt = (
            getattr(self.args, "use_layer_probe_checkpointing", True) and self.training
        )

        def _layer_probe_fn(h, _l):
            cls_l  = self.wordrep.pooling(h, attention_mask)
            wh_l   = torch.bmm(align, h)
            i_l    = self.exit_intent_heads[_l](cls_l)
            s_l    = self.exit_slot_probes[_l](wh_l)
            _, _, _, hard_l, biaffine_l = self._bislu_head(cls_l, wh_l, word_attention_mask)
            return i_l, s_l, hard_l, biaffine_l

        l_int, l_slot, l_bislu_i, l_bislu_s = [], [], [], []
        for l in range(self.num_layers):
            h = hs[l + 1]
            if use_layer_ckpt and h.requires_grad:

                i_l, s_l, hard_l, biaffine_l = torch.utils.checkpoint.checkpoint(
                    _layer_probe_fn, h, l, use_reentrant=False, preserve_rng_state=True,
                )
            else:
                i_l, s_l, hard_l, biaffine_l = _layer_probe_fn(h, l)
            l_int.append(i_l)
            l_slot.append(s_l)
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

    def _true_layer_iter(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Generator[Tuple[int, torch.Tensor], None, None]:

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

        # ── 5. Layer-by-layer iteration ──────────────────────────────────
        for l, layer in enumerate(bm.layers):
            kw = _layer_kwargs_for(
                layer, causal_mask, position_ids, cache_position, position_embeddings
            )
            layer_out = layer(h, **kw)


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

        B      = input_ids.size(0)
        device = input_ids.device
        dummy  = next(self.wordrep.base_model.parameters())
        align  = _build_align(input_ids, words_lengths, device).to(dtype=dummy.dtype)

        # ── Per-sample minimum exit layer from frequency rarity ───────────
        if getattr(self.args, "disable_early_exit", False):

            per_min = torch.full((B,), self.num_layers - 1,
                                 dtype=torch.long, device=device)
        elif self.use_freq_exit and freq_scores is not None:

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

        self.history: Dict[str, List[float]] = {
            "epoch": [], "train_loss": [], "dev_loss": [], "dev_tuning_metric": [],
        }
        self.last_eval_records: List[Dict] = []
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

        opt.zero_grad(set_to_none=True)
        gs = 0
        total_samples_seen = 0
        t_epoch_start      = time.perf_counter()
        accum              = max(self.args.gradient_accumulation_steps, 1)
        max_bad_consec     = getattr(self.args, "max_consecutive_nonfinite", 20)
        bad_consec         = 0
        bad_total          = 0

        for epoch in trange(self.args.num_train_epochs):
            self.model.train()
            ep_loss = 0.0; ep_steps = 0
            t_step  = time.perf_counter()
            micro_in_window = 0
            opt.zero_grad(set_to_none=True)   # drop any partial window from the previous epoch

            for step, batch in enumerate(dl):
                batch_size  = batch[0].size(0)
                freq_scores = batch[6].to(self.device)
                inputs = {
                    "input_ids":           batch[0].to(self.device),
                    "attention_mask":      batch[1].to(self.device),
                    "words_lengths":       batch[2].to(self.device),
                    "word_attention_mask": batch[3].to(self.device),
                }
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

                if accum > 1:
                    loss = loss / accum

                if not torch.isfinite(loss):
                    bad_consec += 1; bad_total += 1
                    logger.warning(
                        "Non-finite LOSS epoch=%d step=%d (consecutive=%d, total=%d). "
                        "Dropping accumulation window.",
                        epoch, step, bad_consec, bad_total)
                    opt.zero_grad(set_to_none=True)
                    micro_in_window = 0
                    if bad_consec >= max_bad_consec:
                        raise RuntimeError(
                            f"Training diverged: {bad_consec} consecutive non-finite "
                            f"losses at epoch={epoch} step={step} "
                            f"(lr={self.args.learning_rate:g}, "
                            f"ee_loss_coef={self.args.ee_loss_coef:g}, "
                            f"bislu_aux_loss_coef={self.args.bislu_aux_loss_coef:g}). "
                            f"Aborting instead of silently continuing with a "
                            f"corrupted model.")
                    continue

                loss.backward()
                ep_loss += loss.item(); ep_steps += 1
                micro_in_window += 1

                if micro_in_window == accum:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm)

                    if not torch.isfinite(grad_norm):
                        bad_consec += 1; bad_total += 1
                        logger.warning(
                            "Non-finite GRAD NORM epoch=%d step=%d (consecutive=%d, "
                            "total=%d). Dropping accumulation window.",
                            epoch, step, bad_consec, bad_total)
                        opt.zero_grad(set_to_none=True)
                        micro_in_window = 0
                        if bad_consec >= max_bad_consec:
                            raise RuntimeError(
                                f"Training diverged: {bad_consec} consecutive "
                                f"non-finite gradients at epoch={epoch} step={step} "
                                f"(lr={self.args.learning_rate:g}). Aborting.")
                        continue

                    opt.step(); sched.step()
                    opt.zero_grad(set_to_none=True)
                    bad_consec      = 0        # only a SUCCESSFUL step resets the counter
                    micro_in_window = 0
                    gs += 1; total_samples_seen += batch_size * accum
                    self.trainer_state.epoch       = epoch
                    self.trainer_state.global_step = gs
                    self.trainer_state.max_steps   = steps
                    self.trainer_state.loss        = ep_loss / max(ep_steps, 1)

                    if _wandb_active:
                        t_now   = time.perf_counter()
                        elapsed = max(t_now - t_step, 1e-6)
                        t_step  = t_now
                        wb.log({
                            "train/loss":              loss.item() * accum,
                            "train/loss_smoothed":     ep_loss / max(ep_steps, 1),
                            "train/learning_rate":     sched.get_last_lr()[0],
                            "train/grad_norm":         grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm),
                            "train/nonfinite_total":   bad_total,
                            "perf/samples_per_sec":    (batch_size * accum) / elapsed,
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
            if bad_total > 0:
                logger.warning("Epoch %d finished with %d total non-finite events so far.",
                               epoch, bad_total)
            if _wandb_active:
                wb.log({"epoch/train_loss": epoch_loss,
                        "epoch/epoch_time_sec": epoch_time,
                        "epoch/epoch": epoch}, step=gs)

            results = self.evaluate("dev", global_step=gs, epoch=epoch)
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(epoch_loss)
            self.history["dev_loss"].append(results.get("loss", float("nan")))
            self.history["dev_tuning_metric"].append(
                results.get(self.args.tuning_metric, float("nan")))
            es(results[self.args.tuning_metric], self.args)
            if es.counter == 0: self.save_model()
            if es.early_stop: logger.info("Early stopping."); break

        wb.finish()

        wb = _WandbDummy()
        _wandb_active = False

    def evaluate(self, mode="dev", global_step: int = 0, epoch: int = 0,
                 capture_samples: bool = False):
        ds = {"dev": self.dev_ds, "test": self.test_ds}.get(mode)
        if ds is None: raise ValueError(f"mode must be dev or test, got {mode!r}")
        logger.info("Eval [%s] %d samples", mode, len(ds))
        dl = self._dl(ds, False)
        self.model.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ev_loss = 0.0
        int_la, int_pa, slot_la, slot_pa, mask_a = [], [], [], [], []
        all_exit_lyrs:   List[int]   = []
        all_freq_scores: List[float] = []
        layer_exit_counts = defaultdict(int)
        ce = nn.CrossEntropyLoss(reduction="mean")
        t_eval_start = time.perf_counter()

        if capture_samples:
            self.last_eval_records = []
        running_idx = 0

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

                if capture_samples:
                    bsz = final_i.size(0)
                    pred_idx = final_i.detach().float().argmax(dim=1).cpu().tolist()
                    gold_idx = il.detach().cpu().argmax(dim=1).tolist()
                    yt_b, yp_b = get_slot_label_lists(
                        sl.detach().cpu(), biaffine.detach().float().cpu(),
                        batch[3].detach().cpu(), self.slot_label_set)
                    fs_b = freq_scores.detach().cpu().tolist()
                    ex_b = exit_lyr_batch.detach().cpu().tolist()
                    for bi in range(bsz):
                        ds_idx = running_idx + bi
                        words = _reconstruct_words(ds, ds_idx)
                        self.last_eval_records.append({
                            "ds_index":       ds_idx,
                            "utterance":      words,
                            "utt_len":        len(words),
                            "rarity_score":   fs_b[bi],
                            "exit_layer":     ex_b[bi],
                            "gold_intent_idx": gold_idx[bi],
                            "pred_intent_idx": pred_idx[bi],
                            "gold_slots":     sorted(yt_b[bi]),
                            "pred_slots":     sorted(yp_b[bi]),
                        })
                    running_idx += bsz

            int_la.append(il.detach().cpu())
            int_pa.append(final_i.detach().cpu())
            slot_la.append(sl.detach().cpu())
            slot_pa.append(biaffine.detach().cpu())
            mask_a.append(batch[3])  # already CPU: word_attention_mask was never .to(device)'d here

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
# 12. REPRODUCIBILITY UTILITIES
# ============================================================

def set_seed(seed: int) -> None:
    """Seed python/numpy/torch (+ CUDA) RNGs for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clone_args(args, **overrides):
    """Deep-copy an argparse.Namespace and apply attribute overrides."""
    new_args = copy.deepcopy(args)
    for k, v in overrides.items():
        setattr(new_args, k, v)
    return new_args


def build_datasets_bundle(args) -> Dict:
    hf_train, hf_dev, hf_test, utt_f, int_f, slot_f, is_instr = load_hf_dataset(
        dataset_name=args.hf_dataset, cache_dir=args.cache_dir or None,
        dev_split_name=args.dev_split, test_split_name=args.test_split,
        train_split_name=args.train_split,
        dev_fraction=args.dev_fraction, test_fraction=args.test_fraction,
    )
    intent_label_set, slot_label_set = extract_label_sets(hf_train, int_f, slot_f, is_instr)
    tokenizer = setup_tokenizer(args.model_name_or_path)

    # --- Create subset for HPO ---
    is_hpo_mode = getattr(args, "mode", "train") in ["hpo", "full", "final"]

    if is_hpo_mode and getattr(args, "hpo_subset_fraction", 1.0) < 1.0:
        # Use subset for HPO trials
        subset_size = int(len(hf_train) * args.hpo_subset_fraction)
        if subset_size < 10:
            logger.warning(f"HPO subset too small ({subset_size}), using 10 samples minimum")
            subset_size = min(10, len(hf_train))

        # Create stratified subset if possible, otherwise random
        hf_train_subset = hf_train.shuffle(seed=getattr(args, "hpo_seed", 42)).select(range(subset_size))
        logger.info(f"HPO subset: Using {subset_size} / {len(hf_train)} samples ({args.hpo_subset_fraction*100:.0f}%)")
        train_for_hpo = hf_train_subset
    else:
        train_for_hpo = hf_train
        logger.info("HPO using full training set")

    freq_index = WordFrequencyIndex(
        smoothing=getattr(args, "freq_smoothing", 0.5),
        min_freq=getattr(args, "freq_min_count", 1),
    )
    # Build frequency index on FULL dataset (important for rarity scoring)
    freq_index.build(hf_train, utt_f, is_instr)

    make_ds = lambda split_data: HFSLUDataset(
        args=args, hf_split=split_data,
        utterance_field=utt_f, intent_field=int_f, slot_field=slot_f,
        intent_label_set=intent_label_set, slot_label_set=slot_label_set,
        tokenizer=tokenizer, is_instruction=is_instr, freq_index=freq_index,
    )

    return {
        "train_ds": make_ds(train_for_hpo),  # ← HPO uses SUBSET
        "dev_ds": make_ds(hf_dev),
        "test_ds": make_ds(hf_test),
        "tokenizer": tokenizer,
        "intent_label_set": intent_label_set,
        "slot_label_set": slot_label_set,
        "freq_index": freq_index,
        "full_train_ds": make_ds(hf_train),  # ← Store full for final training
    }

def _new_trainer(args, bundle) -> "EarlyExitTrainer":
    return EarlyExitTrainer(
        args=args, tokenizer=bundle["tokenizer"],
        train_ds=bundle["train_ds"], dev_ds=bundle["dev_ds"], test_ds=bundle["test_ds"],
        intent_label_set=bundle["intent_label_set"], slot_label_set=bundle["slot_label_set"],
    )


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 13. HYPERPARAMETER OPTIMISATION (Optuna)
# ============================================================

try:
    import optuna
    from optuna.samplers import TPESampler
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    logger.warning("optuna not installed -> HPO mode disabled. "
                    "Run: pip install optuna --break-system-packages")


def _suggest_hpo_params(trial, base_args) -> Dict:

    if getattr(base_args, "freeze_backbone", False):
        lr_low, lr_high = 1e-4, 1e-2
    else:
        lr_low, lr_high = 5e-6, 1e-4

    hp: Dict = {
        "learning_rate":       trial.suggest_float("learning_rate", lr_low, lr_high, log=True),
        "warmup_proportion":   trial.suggest_float("warmup_proportion", 0.03, 0.2),
        "dropout_rate":        trial.suggest_float("dropout_rate", 0.05, 0.3),
        "weight_decay":        trial.suggest_float("weight_decay", 0.0, 0.1),
        "ee_patience":         trial.suggest_int("ee_patience", 1, 6),
        "tau_intent":          trial.suggest_float("tau_intent", 0.01, 0.2, log=True),
        "tau_slot":            trial.suggest_float("tau_slot", 0.02, 0.3, log=True),
        "ee_loss_coef":        trial.suggest_float("ee_loss_coef", 0.05, 0.5),
        "bislu_aux_loss_coef": trial.suggest_float("bislu_aux_loss_coef", 0.05, 0.5),
        "biaffine_dim":        trial.suggest_categorical("biaffine_dim", [64, 128, 192, 256]),
        "hidden_dim_ffw":      trial.suggest_categorical("hidden_dim_ffw", [150, 300, 450, 600]),
    }
    if getattr(base_args, "use_scl", False):
        hp["loss_coef_intent_scl"] = trial.suggest_float("loss_coef_intent_scl", 0.05, 0.5)
        hp["loss_coef_slot_scl"]   = trial.suggest_float("loss_coef_slot_scl", 0.05, 0.5)
    if getattr(base_args, "use_sd", False):
        hp["sd_loss_coef"] = trial.suggest_float("sd_loss_coef", 0.05, 0.5)
    return hp


def _hpo_objective(trial, base_args, bundle: Dict) -> float:
    hp = _suggest_hpo_params(trial, base_args)
    trial_args = _clone_args(base_args, **hp)
    trial_args.num_train_epochs = getattr(base_args, "hpo_epochs", 4)
    trial_args.early_stopping   = getattr(base_args, "hpo_early_stopping", 2)
    trial_args.use_wandb        = False
    trial_args.output_dir       = os.path.join(
        base_args.output_dir, "hpo_trials", f"trial_{trial.number}")

    set_seed(getattr(base_args, "hpo_seed", 42))

    trainer = None
    try:
        trainer = _new_trainer(trial_args, bundle)
        trainer.train()   # may raise RuntimeError (divergence guard) → trial FAILs via catch=
        dev_results = trainer.evaluate("dev")
        score = dev_results.get(trial_args.tuning_metric, float("nan"))
    finally:
        if trainer is not None:
            del trainer
        _cleanup_cuda()

    if not math.isfinite(score):

        raise optuna.TrialPruned(
            f"Non-finite {trial_args.tuning_metric}={score!r} for trial {trial.number}.")
    return score


def run_hpo(base_args, bundle: Dict) -> Optional[Dict]:

    if not _OPTUNA_AVAILABLE:
        logger.error("Cannot run HPO: optuna is not installed.")
        return None

    os.makedirs(base_args.output_dir, exist_ok=True)
    storage_path = os.path.join(base_args.output_dir, "optuna_study.db")
    if os.path.exists(storage_path):
        try:
            conn = sqlite3.connect(storage_path)
            status = conn.execute("PRAGMA integrity_check;").fetchone()
            conn.close()
            if status[0] != "ok":
                logger.warning("Optuna SQLite DB failed integrity_check (%s); "
                               "starting a fresh study DB.", status[0])
                os.remove(storage_path)
        except sqlite3.Error as e:
            logger.warning("SQLite integrity check raised %s; starting fresh.", e)
            if os.path.exists(storage_path):
                os.remove(storage_path)

    study = optuna.create_study(
        study_name=getattr(base_args, "study_name", "bislu_pabee_hpo_v2"),
        storage=f"sqlite:///{storage_path}",
        direction="maximize",
        load_if_exists=True,
        sampler=TPESampler(seed=getattr(base_args, "hpo_seed", 42)),
    )
    n_trials = getattr(base_args, "n_trials", 30)
    study.optimize(
        lambda trial: _hpo_objective(trial, base_args, bundle),
        n_trials=n_trials, catch=(RuntimeError,), gc_after_trial=True,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.error("HPO finished with zero completed trials out of %d requested.", n_trials)
        return None

    best_params = study.best_params
    best_value  = study.best_value
    logger.info("HPO complete: best %s = %.4f | params = %s",
                base_args.tuning_metric, best_value, best_params)

    out_path = os.path.join(base_args.output_dir, "best_hyperparameters.json")
    with open(out_path, "w") as f:
        json.dump({
            "best_value": best_value, "best_params": best_params,
            "tuning_metric": base_args.tuning_metric,
            "n_trials_requested": n_trials, "n_trials_completed": len(completed),
            "n_trials_total": len(study.trials),
        }, f, indent=2)
    logger.info("Saved best hyperparameters -> %s", out_path)
    return best_params


def run_final_train_eval_test(base_args, bundle: Dict, best_params: Optional[Dict] = None):
    final_args = _clone_args(base_args, **(best_params or {}))
    final_args.output_dir = os.path.join(base_args.output_dir, "final_model")
    os.makedirs(final_args.output_dir, exist_ok=True)
    set_seed(getattr(base_args, "seed", 42))

    # --- Use FULL training set for final training ---
    if "full_train_ds" in bundle:
        logger.info("Final training using FULL dataset")
        bundle_for_train = bundle.copy()
        bundle_for_train["train_ds"] = bundle["full_train_ds"]
    else:
        bundle_for_train = bundle

    trainer = _new_trainer(final_args, bundle_for_train)
    trainer.train()
    trainer.load_model()
    test_results = trainer.evaluate("test", capture_samples=True)

    with open(os.path.join(final_args.output_dir, "best_hp_used.json"), "w") as f:
        json.dump(best_params or {}, f, indent=2)
    with open(os.path.join(final_args.output_dir, "final_test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    return trainer, test_results


# ============================================================
# 14. ABLATION STUDY (8 experiment groups incl. pairwise combinations)
# ============================================================

ABLATION_COMPONENTS: List[str] = ["freq_pabee", "soft_slot", "scl", "sd"]


def _component_flags(active: Set[str]) -> Dict[str, bool]:

    scl = "scl" in active
    soft_slot_explicit = "soft_slot" in active
    soft_slot = soft_slot_explicit or scl
    return {
        "disable_early_exit": False,
        "use_freq_exit":      "freq_pabee" in active,
        "use_soft_slot":      soft_slot,
        "use_scl":            scl,
        "use_sd":             "sd" in active,
        "implied_soft_slot":  scl and not soft_slot_explicit,
    }


ABLATION_EXPERIMENTS: Dict[str, Dict] = {
    # Exp 1: no early exit, no SCL, no SD, no soft slot — standard BiSLU.
    "E1_baseline": {
        "disable_early_exit": True, "use_freq_exit": False,
        "use_soft_slot": False, "use_scl": False, "use_sd": False,
        "implied_soft_slot": False,
    },
    # Exp 2: PABEE with a fixed (non frequency-adaptive) exit threshold.
    "E2_pabee_fixed": _component_flags(set()),
    # Exp 3: frequency-adaptive PABEE (per-sample min-exit-layer from rarity).
    "E3_pabee_freq_adaptive": _component_flags({"freq_pabee"}),
    # Exp 4: soft slot-to-intent attention feedback only.
    "E4_soft_slot": _component_flags({"soft_slot"}),
    # Exp 5: supervised contrastive learning (requires soft slot infra).
    "E5_scl": _component_flags({"soft_slot", "scl"}),
    # Exp 6: self-distillation (deep -> shallow KL).
    "E6_sd": _component_flags({"sd"}),
    # Exp 7: full model — freq-adaptive PABEE + SCL + SD + soft slot.
    "E7_full_model": _component_flags({"freq_pabee", "soft_slot", "scl", "sd"}),
}


for _a, _b in itertools.combinations(ABLATION_COMPONENTS, 2):
    ABLATION_EXPERIMENTS[f"E8_{_a}+{_b}"] = _component_flags({_a, _b})


def run_ablation_study(base_args, bundle: Dict,
                       seeds: Tuple[int, ...] = (42, 43, 44),
                       experiments: Optional[List[str]] = None,
                       best_params: Optional[Dict] = None) -> "pd.DataFrame":

    experiments = experiments or list(ABLATION_EXPERIMENTS.keys())
    records: List[Dict] = []
    for exp_name in experiments:
        flags = ABLATION_EXPERIMENTS[exp_name]
        for seed in seeds:
            exp_args = _clone_args(base_args, **(best_params or {}))
            for k, v in flags.items():
                if k != "implied_soft_slot":
                    setattr(exp_args, k, v)
            exp_args.output_dir = os.path.join(
                base_args.output_dir, "ablation", exp_name, f"seed{seed}")
            exp_args.use_wandb = False
            os.makedirs(exp_args.output_dir, exist_ok=True)
            set_seed(seed)
            logger.info("=== Ablation %s | seed=%d | flags=%s ===", exp_name, seed, flags)
       
            trainer = None
            try:
                trainer = _new_trainer(exp_args, bundle)
                trainer.train()
                trainer.load_model()
                test_res = trainer.evaluate("test")
                records.append({
                    "experiment": exp_name, "seed": seed, "failed": False,
                    **flags, **test_res,
                })
            except RuntimeError as e:
                logger.error("Ablation %s seed=%d failed: %s", exp_name, seed, e)
                records.append({
                    "experiment": exp_name, "seed": seed, "failed": True,
                    "error": str(e), **flags,
                })
            finally:

                if trainer is not None:
                    del trainer
                _cleanup_cuda()

    df = pd.DataFrame(records)
    os.makedirs(base_args.output_dir, exist_ok=True)
    out_csv = os.path.join(base_args.output_dir, "ablation_results.csv")
    df.to_csv(out_csv, index=False)


def run_cross_dataset_difficulty(base_args, dataset_names: List[str],
                                 experiments: Optional[List[str]] = None,
                                 seeds: Tuple[int, ...] = (42,)) -> "pd.DataFrame":

    experiments = experiments or ["E1_baseline", "E3_pabee_freq_adaptive", "E7_full_model"]
    rows: List[Dict] = []
    for ds_name in dataset_names:
        logger.info("=== Cross-dataset difficulty: %s ===", ds_name)
        ds_args = _clone_args(base_args, hf_dataset=ds_name)
        try:
            ds_bundle = build_datasets_bundle(ds_args)
        except Exception as e:  # dataset-level failures should not abort the sweep
            logger.error("Failed to load dataset %s: %s", ds_name, e)
            continue
        sub_df = run_ablation_study(
            _clone_args(ds_args, output_dir=os.path.join(
                base_args.output_dir, "cross_dataset", ds_name.replace("/", "_"))),
            ds_bundle, seeds=seeds, experiments=experiments,
        )
        sub_df["dataset"] = ds_name
        rows.append(sub_df)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not df.empty:
        df.to_csv(os.path.join(base_args.output_dir, "cross_dataset_difficulty.csv"), index=False)
    return df


# ============================================================
# 15. STATISTICAL SIGNIFICANCE TESTING
# ============================================================

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled sample standard deviation."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_var = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / max(n1 + n2 - 2, 1)
    pooled_std = math.sqrt(pooled_var)
    if pooled_std == 0.0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def _bh_fdr(pvalues: List[float], alpha: float = 0.05) -> List[bool]:

    m = len(pvalues)
    if m == 0:
        return []
    pvals = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = (np.arange(1, m + 1) / m) * alpha
    below  = ranked <= thresh
    reject_sorted = np.zeros(m, dtype=bool)
    if below.any():
        max_k = int(np.max(np.nonzero(below)[0]))
        reject_sorted[: max_k + 1] = True
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject.tolist()


def run_statistical_tests(ablation_df: "pd.DataFrame", baseline_exp: str = "E1_baseline",
                          metric: str = "mean_intent_slot") -> "pd.DataFrame":

    from scipy import stats

    clean_df = ablation_df[~ablation_df.get("failed", False).astype(bool)] \
        if "failed" in ablation_df.columns else ablation_df
    base_vals = clean_df.loc[clean_df.experiment == baseline_exp, metric].dropna().values

    if len(base_vals) < 2:
        logger.warning("Baseline experiment %s has <2 valid seeds for metric %s; "
                       "statistical tests cannot be computed.", baseline_exp, metric)

    rows: List[Dict] = []
    for exp_name, grp in clean_df.groupby("experiment"):
        if exp_name == baseline_exp:
            continue
        vals = grp[metric].dropna().values
        if len(vals) < 2 or len(base_vals) < 2:
            rows.append({"experiment": exp_name, "n": len(vals),
                        "note": "n<2 seeds: significance test not computable"})
            continue
        paired = len(vals) == len(base_vals)
        if paired:
            t_stat, t_p = stats.ttest_rel(vals, base_vals)
            try:
                w_stat, w_p = stats.wilcoxon(vals, base_vals)
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")
        else:
            t_stat, t_p = stats.ttest_ind(vals, base_vals, equal_var=False)
            w_stat, w_p = float("nan"), float("nan")
        rows.append({
            "experiment":     exp_name, "n": len(vals), "paired": paired,
            "mean":           float(np.mean(vals)),
            "std":            float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "baseline_mean":  float(np.mean(base_vals)),
            "delta":          float(np.mean(vals) - np.mean(base_vals)),
            "t_stat":         float(t_stat), "t_pvalue": float(t_p),
            "wilcoxon_stat":  float(w_stat), "wilcoxon_pvalue": float(w_p),
            "cohens_d":       _cohens_d(vals, base_vals),
        })

    res_df = pd.DataFrame(rows)
    if not res_df.empty and "t_pvalue" in res_df.columns:
        res_df["t_pvalue_fdr_reject_at_0.05"] = _bh_fdr(
            res_df["t_pvalue"].fillna(1.0).tolist(), alpha=0.05)
    logger.warning("Statistical tests computed with n=%d seed(s) per arm — "
                   "treat p-values as indicative, not confirmatory, unless "
                   "n is increased.", len(base_vals))
    return res_df


# ============================================================
# 16. SENSITIVITY ANALYSIS (one-factor-at-a-time)
# ============================================================

SENSITIVITY_GRID_FULL_FT: Dict[str, List] = {
    "learning_rate":       [5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 3e-4],
    "dropout_rate":        [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    "ee_patience":         [1, 2, 3, 4, 5, 6],
    "tau_intent":          [0.01, 0.03, 0.05, 0.08, 0.12, 0.2],
    "tau_slot":            [0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
    "ee_loss_coef":        [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    "bislu_aux_loss_coef": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    "biaffine_dim":        [64, 128, 192, 256],
    "hidden_dim_ffw":      [150, 300, 450, 600],
    "weight_decay":        [0.0, 0.01, 0.05, 0.1],
    "warmup_proportion":   [0.0, 0.05, 0.1, 0.15, 0.2],
}

SENSITIVITY_GRID_FROZEN: Dict[str, List] = {
    **{k: v for k, v in SENSITIVITY_GRID_FULL_FT.items() if k != "learning_rate"},
    "learning_rate": [5e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
}


def run_sensitivity_analysis(base_args, bundle: Dict,
                             center_params: Optional[Dict] = None
                             ) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    center = dict(center_params or {})
    grid_source = (SENSITIVITY_GRID_FROZEN if getattr(base_args, "freeze_backbone", False)
                   else SENSITIVITY_GRID_FULL_FT)
    records: List[Dict] = []
    for param, grid in grid_source.items():
        for val in grid:
            trial_args = _clone_args(base_args, **center)
            setattr(trial_args, param, val)
            trial_args.num_train_epochs = getattr(base_args, "sensitivity_epochs", 3)
            trial_args.early_stopping   = getattr(base_args, "sensitivity_early_stopping", 2)
            trial_args.use_wandb        = False
            trial_args.output_dir       = os.path.join(
                base_args.output_dir, "sensitivity", param, str(val))
            set_seed(getattr(base_args, "seed", 42))
            score = float("nan")
            trainer = None
            try:
                trainer = _new_trainer(trial_args, bundle)
                trainer.train()
                dev_res = trainer.evaluate("dev")
                score = dev_res.get(base_args.tuning_metric, float("nan"))
            except RuntimeError as e:
                logger.error("Sensitivity %s=%s failed: %s", param, val, e)
            finally:
            
                if trainer is not None:
                    del trainer
                _cleanup_cuda()
            records.append({"param": param, "value": val, "score": score})

    df = pd.DataFrame(records)
    os.makedirs(base_args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(base_args.output_dir, "sensitivity_results.csv"), index=False)

    from scipy import stats as _stats
    corr_rows: List[Dict] = []
    for param, grp in df.groupby("param"):
        clean = grp.dropna(subset=["score"])
        if len(clean) < 3:
            corr_rows.append({"param": param, "spearman_rho": float("nan"),
                             "spearman_p": float("nan"), "n": len(clean)})
            continue
        rho, p = _stats.spearmanr(clean["value"].astype(float), clean["score"].astype(float))
        corr_rows.append({"param": param, "spearman_rho": rho, "spearman_p": p, "n": len(clean)})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(base_args.output_dir, "sensitivity_correlations.csv"), index=False)
    return df, corr_df


# ============================================================
# 17. ERROR ANALYSIS / CASE STUDIES / CORRELATION / STRATIFICATION
# ============================================================

def error_analysis(records: List[Dict], intent_label_set: List[str]) -> Dict:

    n = len(records)
    if n == 0:
        return {}
    confusion: Dict[Tuple[str, str], int] = defaultdict(int)
    for r in records:
        if r["gold_intent_idx"] != r["pred_intent_idx"]:
            confusion[(intent_label_set[r["gold_intent_idx"]],
                      intent_label_set[r["pred_intent_idx"]])] += 1

    slot_categories: Dict[str, int] = defaultdict(int)
    for r in records:
        gold, pred = set(map(tuple, r["gold_slots"])), set(map(tuple, r["pred_slots"]))
        slot_categories["exact_match"] += len(gold & pred)
        for g in gold - pred:
            gname, gs, ge = g
            overlap = [p for p in pred if not (p[2] < gs or p[1] > ge)]
            if overlap:
                slot_categories["wrong_type_or_boundary"] += 1
            else:
                slot_categories["missed_entity"] += 1
        for p in pred - gold:
            pname, ps, pe = p
            overlap = [g for g in gold if not (g[2] < ps or g[1] > pe)]
            if not overlap:
                slot_categories["spurious_entity"] += 1

    return {
        "n_samples":            n,
        "intent_error_rate":    sum(1 for r in records
                                    if r["gold_intent_idx"] != r["pred_intent_idx"]) / n,
        "intent_confusion_pairs": {f"{g}->{p}": c for (g, p), c in confusion.items()},
        "slot_error_categories": dict(slot_categories),
    }


def select_case_studies(records: List[Dict], intent_label_set: List[str],
                        n_per_bucket: int = 3, seed: int = 0) -> List[Dict]:
    """Qualitative examples stratified by (early/late exit) x (correct/incorrect)."""
    if not records:
        return []
    rng = random.Random(seed)
    max_layer = max(r["exit_layer"] for r in records)
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        correct = r["gold_intent_idx"] == r["pred_intent_idx"]
        early = r["exit_layer"] < max(max_layer * 0.5, 1e-9)
        key = f"{'early' if early else 'late'}_exit_{'correct' if correct else 'incorrect'}"
        buckets[key].append(r)

    cases = []
    for key, pool in buckets.items():
        for r in rng.sample(pool, min(n_per_bucket, len(pool))):
            cases.append({
                "bucket":       key,
                "utterance":    " ".join(r["utterance"]),
                "gold_intent":  intent_label_set[r["gold_intent_idx"]],
                "pred_intent":  intent_label_set[r["pred_intent_idx"]],
                "exit_layer":   r["exit_layer"],
                "rarity_score": round(r["rarity_score"], 3),
            })
    return cases


def ablation_component_correlation(ablation_df: "pd.DataFrame",
                                   metric: str = "mean_intent_slot") -> "pd.DataFrame":

    comp_cols = [c for c in ("use_freq_exit", "use_soft_slot", "use_scl", "use_sd")
                if c in ablation_df.columns]
    df = ablation_df.dropna(subset=[metric]).copy()
    if df.empty or not comp_cols:
        return pd.DataFrame()
    for c in comp_cols:
        df[c] = df[c].astype(float)
    return df[comp_cols + [metric]].corr(method="pearson")


def stratify_by_length(records: List[Dict], n_bins: int = 5) -> "pd.DataFrame":
    if not records:
        return pd.DataFrame()
    lens = np.array([r["utt_len"] for r in records], dtype=float)
    correct = np.array([r["gold_intent_idx"] == r["pred_intent_idx"] for r in records], dtype=float)
    edges = np.unique(np.quantile(lens, np.linspace(0, 1, n_bins + 1)))
    bin_idx = np.digitize(lens, edges[1:-1], right=True)
    rows = []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        rows.append({"length_bin": f"[{edges[b]:.0f},{edges[b+1]:.0f}]",
                    "n": int(mask.sum()), "intent_acc": float(correct[mask].mean()),
                    "mean_utt_len": float(lens[mask].mean())})
    return pd.DataFrame(rows)


def stratify_by_rarity(records: List[Dict], n_bins: int = 5) -> "pd.DataFrame":
    if not records:
        return pd.DataFrame()
    rar = np.array([r["rarity_score"] for r in records], dtype=float)
    correct = np.array([r["gold_intent_idx"] == r["pred_intent_idx"] for r in records], dtype=float)
    exit_l = np.array([r["exit_layer"] for r in records], dtype=float)
    edges = np.unique(np.quantile(rar, np.linspace(0, 1, n_bins + 1)))
    bin_idx = np.digitize(rar, edges[1:-1], right=True)
    rows = []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        rows.append({"rarity_bin": f"[{edges[b]:.2f},{edges[b+1]:.2f}]",
                    "n": int(mask.sum()), "intent_acc": float(correct[mask].mean()),
                    "mean_exit_layer": float(exit_l[mask].mean())})
    return pd.DataFrame(rows)

# ============================================================
# 18. VISUALIZATIONS (10 figures for the paper)
# ============================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fig_dir(output_dir: str) -> str:
    d = os.path.join(output_dir, "figures")
    os.makedirs(d, exist_ok=True)
    return d


def fig_sensitivity_heatmap(sens_df: "pd.DataFrame", output_dir: str) -> Optional[str]:
    """Figure 1: heatmap of normalized sensitivity score per parameter x quantile position."""
    if sens_df is None or sens_df.empty:
        logger.warning("fig_sensitivity_heatmap: no sensitivity data; skipping.")
        return None
    params = list(sens_df["param"].unique())
    n_pos  = max(len(sens_df[sens_df.param == p]) for p in params)
    grid   = np.full((len(params), n_pos), np.nan)
    for i, p in enumerate(params):
        vals = sens_df.loc[sens_df.param == p].sort_values("value")["score"].to_numpy()
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        lo, hi = finite.min(), finite.max()
        norm = (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)
        grid[i, :len(norm)] = norm

    fig, ax = plt.subplots(figsize=(1.1 * n_pos + 3, 0.5 * len(params) + 2))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(params))); ax.set_yticklabels(params)
    ax.set_xticks(range(n_pos)); ax.set_xticklabels([f"v{j+1}" for j in range(n_pos)])
    ax.set_xlabel("Grid position (low -> high value)")
    ax.set_title("Hyperparameter Sensitivity (normalized dev tuning metric)")
    fig.colorbar(im, ax=ax, label="normalized score")
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "01_sensitivity_heatmap.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_pareto_front(ablation_df: "pd.DataFrame", output_dir: str,
                     metric: str = "mean_intent_slot") -> Optional[str]:
    """Figure 2: accuracy vs. layer savings, with the Pareto-optimal frontier highlighted."""
    if ablation_df is None or ablation_df.empty:
        logger.warning("fig_pareto_front: no ablation data; skipping.")
        return None
    df = ablation_df.dropna(subset=[metric, "layer_savings_pct"])
    if df.empty:
        return None
    agg = df.groupby("experiment").agg(
        acc=(metric, "mean"), savings=("layer_savings_pct", "mean")).reset_index()

    pts = agg[["savings", "acc"]].to_numpy()
    order = np.argsort(-pts[:, 1])  # sort by accuracy desc
    pareto_mask = np.zeros(len(pts), dtype=bool)
    best_savings_so_far = -np.inf
    for idx in order:
        if pts[idx, 0] >= best_savings_so_far:
            pareto_mask[idx] = True
            best_savings_so_far = pts[idx, 0]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(agg.loc[~pareto_mask, "savings"], agg.loc[~pareto_mask, "acc"],
              c="gray", alpha=0.6, label="dominated")
    ax.scatter(agg.loc[pareto_mask, "savings"], agg.loc[pareto_mask, "acc"],
              c="crimson", s=80, label="Pareto-optimal", zorder=3)
    pf = agg.loc[pareto_mask].sort_values("savings")
    ax.plot(pf["savings"], pf["acc"], "--", c="crimson", alpha=0.7)
    for _, row in agg.iterrows():
        ax.annotate(row["experiment"], (row["savings"], row["acc"]),
                   fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Layer savings (fraction of depth skipped)")
    ax.set_ylabel(metric)
    ax.set_title("Pareto Front: Accuracy vs. Layer Savings")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "02_pareto_front.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_boxplot_seeds(ablation_df: "pd.DataFrame", output_dir: str,
                      metric: str = "mean_intent_slot") -> Optional[str]:
    """Figure 3: distribution of test performance across seeds, per ablation experiment."""
    if ablation_df is None or ablation_df.empty:
        logger.warning("fig_boxplot_seeds: no ablation data; skipping.")
        return None
    df = ablation_df.dropna(subset=[metric])
    experiments = sorted(df["experiment"].unique())
    data = [df.loc[df.experiment == e, metric].to_numpy() for e in experiments]

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(experiments)), 6))
    ax.boxplot(data, labels=experiments, showmeans=True)
    ax.set_ylabel(metric)
    ax.set_title(f"Performance Across Seeds (n={df.groupby('experiment').size().max()})")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "03_boxplot_seeds.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_bar_ablation(ablation_df: "pd.DataFrame", output_dir: str,
                     metric: str = "mean_intent_slot") -> Optional[str]:
    """Figure 4: bar chart of mean +/- std per ablation experiment."""
    if ablation_df is None or ablation_df.empty:
        logger.warning("fig_bar_ablation: no ablation data; skipping.")
        return None
    df = ablation_df.dropna(subset=[metric])
    agg = df.groupby("experiment")[metric].agg(["mean", "std"]).reset_index()
    agg = agg.sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(agg)), 6))
    ax.bar(agg["experiment"], agg["mean"], yerr=agg["std"].fillna(0.0), capsize=4,
          color="steelblue", edgecolor="black")
    ax.set_ylabel(metric)
    ax.set_title("Ablation Study Results (mean +/- std across seeds)")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "04_bar_ablation.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_scatter_rarity_exit(records: List[Dict], output_dir: str) -> Optional[str]:
    """Figure 5: word-rarity score vs. exit layer, with Pearson r annotated."""
    if not records:
        logger.warning("fig_scatter_rarity_exit: no per-sample records; skipping.")
        return None
    rar = np.array([r["rarity_score"] for r in records])
    exl = np.array([r["exit_layer"] for r in records], dtype=float)
    r = float(np.corrcoef(rar, exl)[0, 1]) if len(rar) > 2 else float("nan")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(rar, exl, alpha=0.3, s=12, c="teal")
    if len(rar) > 2 and np.isfinite(r):
        m, b = np.polyfit(rar, exl, 1)
        xs = np.linspace(rar.min(), rar.max(), 50)
        ax.plot(xs, m * xs + b, "r--", label=f"linear fit (Pearson r={r:.3f})")
        ax.legend()
    ax.set_xlabel("Word-rarity score (0=frequent, 1=rare)")
    ax.set_ylabel("Exit layer")
    ax.set_title("Rarity vs. Exit Layer")
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "05_scatter_rarity_exit.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_learning_curves(history: Dict[str, List[float]], output_dir: str) -> Optional[str]:
    """Figure 6: train loss / dev loss / dev tuning-metric vs. epoch."""
    if not history or not history.get("epoch"):
        logger.warning("fig_learning_curves: empty history; skipping.")
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history["epoch"], history["train_loss"], label="train loss", marker="o")
    ax1.plot(history["epoch"], history["dev_loss"], label="dev loss", marker="s")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(); ax1.set_title("Loss")
    ax2.plot(history["epoch"], history["dev_tuning_metric"], label="dev tuning metric",
             marker="^", color="darkorange")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("tuning metric"); ax2.legend()
    ax2.set_title("Dev Tuning Metric")
    fig.suptitle("Learning Curves")
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "06_learning_curves.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_confusion_matrix(records: List[Dict], intent_label_set: List[str],
                         output_dir: str, max_labels: int = 30) -> Optional[str]:
    """Figure 7: intent confusion matrix (assumes single-intent-per-utterance eval)."""
    if not records:
        logger.warning("fig_confusion_matrix: no per-sample records; skipping.")
        return None
    n_lbl = len(intent_label_set)
    mat = np.zeros((n_lbl, n_lbl), dtype=int)
    for r in records:
        mat[r["gold_intent_idx"], r["pred_intent_idx"]] += 1

    if n_lbl > max_labels:
        # Keep the most frequent gold labels only, to keep the figure legible.
        freq = mat.sum(axis=1)
        keep = np.argsort(-freq)[:max_labels]
        mat = mat[np.ix_(keep, keep)]
        labels = [intent_label_set[i] for i in keep]
    else:
        labels = intent_label_set

    fig, ax = plt.subplots(figsize=(0.35 * len(labels) + 4, 0.35 * len(labels) + 4))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predicted intent"); ax.set_ylabel("Gold intent")
    ax.set_title("Intent Confusion Matrix" + (" (top-frequency subset)" if n_lbl > max_labels else ""))
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "07_confusion_matrix.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_table_hyperparams(best_params: Dict, output_dir: str) -> Optional[str]:
    """Figure 8: rendered table of optimal hyperparameter values."""
    if not best_params:
        logger.warning("fig_table_hyperparams: no best_params provided; skipping.")
        return None
    rows = [[k, f"{v:.6g}" if isinstance(v, float) else str(v)] for k, v in sorted(best_params.items())]
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(rows) + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["Hyperparameter", "Optimal value"],
                  loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
    ax.set_title("HPO-Selected Hyperparameter Values", pad=20)
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "08_table_hyperparameters.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_table_significance(sig_df: "pd.DataFrame", output_dir: str) -> Optional[str]:
    """Figure 9: rendered table of statistical-significance test results."""
    if sig_df is None or sig_df.empty:
        logger.warning("fig_table_significance: no significance results; skipping.")
        return None
    cols = [c for c in ("experiment", "n", "mean", "std", "delta", "t_pvalue",
                        "wilcoxon_pvalue", "cohens_d", "t_pvalue_fdr_reject_at_0.05")
            if c in sig_df.columns]
    disp = sig_df[cols].copy()
    for c in disp.columns:
        if disp[c].dtype == float:
            disp[c] = disp[c].map(lambda x: f"{x:.4g}" if pd.notna(x) else "NA")

    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 2, 0.4 * len(disp) + 2))
    ax.axis("off")
    tbl = ax.table(cellText=disp.values.tolist(), colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    ax.set_title("Statistical Significance vs. Baseline (n=%d/arm; see caveat in log)"
                % (int(sig_df["n"].max()) if "n" in sig_df.columns and not sig_df.empty else 0),
                pad=20)
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "09_table_significance.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_architecture_overview(output_dir: str) -> str:

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    ax.set_xlim(0, 11); ax.set_ylim(0, 6)

    def box(x, y, w, h, text, color="#dbe9f6"):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, wrap=True)

    box(0.3, 2.2, 1.6, 1.6, "Token\nEmbeddings")
    for i in range(4):
        box(2.2 + i * 1.5, 2.2, 1.2, 1.6, f"Decoder\nLayer {i+1}" + ("\n(...)" if i == 3 else ""))
    box(8.3, 2.2, 1.4, 1.6, "Final\nHidden States")

    for i in range(4):
        box(2.2 + i * 1.5, 0.2, 1.2, 1.2, "PABEE\nintent/slot\nprobe", color="#f6e0b5")
    box(0.3, 4.4, 1.9, 1.2, "Frequency Index\n(rarity score)\n-> per-sample\nmin exit layer", color="#f6b5b5")
    box(4.7, 4.4, 2.2, 1.2, "Soft Slot Feedback\n(slot probs -> intent ctx)", color="#c6e6c6")
    box(7.2, 4.4, 1.9, 1.2, "SCL\n(dropout views,\ncontrastive loss)", color="#c6e6c6")
    box(9.3, 4.4, 1.6, 1.2, "Self-Distillation\n(deep->shallow KL)", color="#c6e6c6")

    ax.annotate("", xy=(1.2, 4.4), xytext=(1.1, 3.0),
               arrowprops=dict(arrowstyle="->", color="crimson"))
    ax.annotate("", xy=(5.5, 4.4), xytext=(5.0, 3.8),
               arrowprops=dict(arrowstyle="->", color="darkgreen"))
    ax.annotate("", xy=(8.0, 4.4), xytext=(6.8, 3.0),
               arrowprops=dict(arrowstyle="->", color="darkgreen"))
    ax.annotate("", xy=(9.8, 4.4), xytext=(8.8, 3.0),
               arrowprops=dict(arrowstyle="->", color="darkgreen"))

    ax.set_title("BiSLU + PABEE Architecture: Component Attachment Points (schematic)")
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "10_architecture_overview.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def generate_all_figures(output_dir: str, sens_df=None, ablation_df=None, records=None,
                         history=None, intent_label_set=None, best_params=None,
                         sig_df=None) -> Dict[str, Optional[str]]:
    """Call every figure function, tolerating missing inputs (each returns None if skipped)."""
    paths: Dict[str, Optional[str]] = {}
    paths["sensitivity_heatmap"] = fig_sensitivity_heatmap(sens_df, output_dir) if sens_df is not None else None
    paths["pareto_front"]        = fig_pareto_front(ablation_df, output_dir) if ablation_df is not None else None
    paths["boxplot_seeds"]       = fig_boxplot_seeds(ablation_df, output_dir) if ablation_df is not None else None
    paths["bar_ablation"]        = fig_bar_ablation(ablation_df, output_dir) if ablation_df is not None else None
    paths["scatter_rarity_exit"] = fig_scatter_rarity_exit(records, output_dir) if records is not None else None
    paths["learning_curves"]     = fig_learning_curves(history, output_dir) if history is not None else None
    paths["confusion_matrix"]    = (fig_confusion_matrix(records, intent_label_set, output_dir)
                                    if records is not None and intent_label_set is not None else None)
    paths["table_hyperparams"]   = fig_table_hyperparams(best_params, output_dir) if best_params is not None else None
    paths["table_significance"]  = fig_table_significance(sig_df, output_dir) if sig_df is not None else None
    paths["architecture"]        = fig_architecture_overview(output_dir)
    return paths

# ============================================================
# 19. ORCHESTRATION: FULL RESEARCH PIPELINE
# ============================================================

def run_full_pipeline(args) -> Dict:

    summary: Dict = {"stages_completed": [], "stages_failed": []}
    os.makedirs(args.output_dir, exist_ok=True)
    bundle = build_datasets_bundle(args)

    # ---- 1. HPO -------------------------------------------------------
    best_params: Optional[Dict] = None
    if getattr(args, "run_hpo", True) and _OPTUNA_AVAILABLE:
        try:
            best_params = run_hpo(args, bundle)
            summary["stages_completed"].append("hpo")
        except Exception as e:
            logger.error("HPO stage failed: %s", e)
            summary["stages_failed"].append({"stage": "hpo", "error": str(e)})
    else:
        logger.info("Skipping HPO stage (run_hpo=%s, optuna_available=%s).",
                    getattr(args, "run_hpo", True), _OPTUNA_AVAILABLE)

    # ---- 2. Final train / eval / test with best hyperparameters -------
    final_trainer, test_results, history, records = None, {}, {}, []
    try:
        final_trainer, test_results = run_final_train_eval_test(args, bundle, best_params)
        history = final_trainer.history
        records = final_trainer.last_eval_records
        summary["stages_completed"].append("final_train_eval_test")
        summary["final_test_results"] = test_results
    except Exception as e:
        logger.error("Final train/eval/test stage failed: %s", e)
        summary["stages_failed"].append({"stage": "final_train_eval_test", "error": str(e)})

    # ---- 3. Ablation study ---------------------------------------------
    ablation_df = pd.DataFrame()
    try:
        seeds = tuple(getattr(args, "seeds", [42, 43, 44]))
        ablation_df = run_ablation_study(args, bundle, seeds=seeds, best_params=best_params)
        summary["stages_completed"].append("ablation")
    except Exception as e:
        logger.error("Ablation stage failed: %s", e)
        summary["stages_failed"].append({"stage": "ablation", "error": str(e)})

    # ---- 4. Statistical significance -----------------------------------
    sig_df = pd.DataFrame()
    if not ablation_df.empty:
        try:
            sig_df = run_statistical_tests(
                ablation_df, baseline_exp=getattr(args, "baseline_exp", "E1_baseline"),
                metric=args.tuning_metric)
            sig_df.to_csv(os.path.join(args.output_dir, "statistical_significance.csv"), index=False)
            summary["stages_completed"].append("statistical_tests")
        except Exception as e:
            logger.error("Statistical testing stage failed: %s", e)
            summary["stages_failed"].append({"stage": "statistical_tests", "error": str(e)})

    # ---- 5. Sensitivity analysis ----------------------------------------
    sens_df, corr_df = pd.DataFrame(), pd.DataFrame()
    try:
        sens_df, corr_df = run_sensitivity_analysis(args, bundle, center_params=best_params)
        summary["stages_completed"].append("sensitivity")
    except Exception as e:
        logger.error("Sensitivity analysis stage failed: %s", e)
        summary["stages_failed"].append({"stage": "sensitivity", "error": str(e)})

    # ---- 6. Additional analyses ------------------------------------------
    analyses: Dict = {}
    try:
        if records:
            analyses["error_analysis"] = error_analysis(records, bundle["intent_label_set"])
            analyses["case_studies"] = select_case_studies(records, bundle["intent_label_set"])
            len_df = stratify_by_length(records)
            rar_df = stratify_by_rarity(records)
            len_df.to_csv(os.path.join(args.output_dir, "length_analysis.csv"), index=False)
            rar_df.to_csv(os.path.join(args.output_dir, "rarity_analysis.csv"), index=False)
        if not ablation_df.empty:
            comp_corr = ablation_component_correlation(ablation_df, metric=args.tuning_metric)
            comp_corr.to_csv(os.path.join(args.output_dir, "ablation_component_correlation.csv"))
        with open(os.path.join(args.output_dir, "additional_analyses.json"), "w") as f:
            json.dump(analyses, f, indent=2, default=str)
        summary["stages_completed"].append("additional_analyses")
    except Exception as e:
        logger.error("Additional analyses stage failed: %s", e)
        summary["stages_failed"].append({"stage": "additional_analyses", "error": str(e)})

    # ---- 7. Cross-dataset difficulty (optional) ---------------------------
    if getattr(args, "cross_datasets", None):
        try:
            run_cross_dataset_difficulty(args, args.cross_datasets,
                                         seeds=tuple(getattr(args, "seeds", [42])[:1]))
            summary["stages_completed"].append("cross_dataset_difficulty")
        except Exception as e:
            logger.error("Cross-dataset difficulty stage failed: %s", e)
            summary["stages_failed"].append({"stage": "cross_dataset_difficulty", "error": str(e)})

    # ---- 8. Figures ----------------------------------------------------
    try:
        fig_paths = generate_all_figures(
            args.output_dir, sens_df=sens_df, ablation_df=ablation_df, records=records,
            history=history, intent_label_set=bundle.get("intent_label_set"),
            best_params=best_params, sig_df=sig_df,
        )
        summary["figures"] = fig_paths
        summary["stages_completed"].append("figures")
    except Exception as e:
        logger.error("Figure generation stage failed: %s", e)
        summary["stages_failed"].append({"stage": "figures", "error": str(e)})

    with open(os.path.join(args.output_dir, "pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Full pipeline finished. Completed=%s Failed=%s",
               summary["stages_completed"], [s["stage"] for s in summary["stages_failed"]])
    return summary


# ============================================================
# 20. MAIN + ARGPARSE
# ============================================================

def _setup_device(args):
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


def main(args):

    _setup_device(args)
    mode = getattr(args, "mode", "train")

    if mode == "train":
        set_seed(getattr(args, "seed", 42))   
        init_wandb(args)
        bundle = build_datasets_bundle(args)
        trainer = EarlyExitTrainer(
            args=args, tokenizer=bundle["tokenizer"],
            train_ds=bundle["train_ds"] if args.do_train else None,
            dev_ds=bundle["dev_ds"], test_ds=bundle["test_ds"],
            intent_label_set=bundle["intent_label_set"], slot_label_set=bundle["slot_label_set"],
        )
        if args.do_train: trainer.train()
        if args.do_eval:  trainer.load_model(); trainer.evaluate("test")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    if mode == "hpo":
        bundle = build_datasets_bundle(args)
        run_hpo(args, bundle)

    elif mode == "final":
        bundle = build_datasets_bundle(args)
        best_params = run_hpo(args, bundle) if getattr(args, "run_hpo", True) else None
        run_final_train_eval_test(args, bundle, best_params)

    elif mode == "ablation":
        bundle = build_datasets_bundle(args)
        best_params = None
        if getattr(args, "best_hp_json", ""):
            with open(args.best_hp_json) as f:
                best_params = json.load(f).get("best_params")
        experiments = getattr(args, "experiments", None) or None
        run_ablation_study(args, bundle, seeds=tuple(args.seeds), 
                           experiments=experiments, best_params=best_params)

    elif mode == "sensitivity":
        bundle = build_datasets_bundle(args)
        best_params = None
        if getattr(args, "best_hp_json", ""):
            with open(args.best_hp_json) as f:
                best_params = json.load(f).get("best_params")
        run_sensitivity_analysis(args, bundle, center_params=best_params)

    elif mode == "stats":
        if not getattr(args, "ablation_csv", ""):
            raise ValueError("--mode stats requires --ablation_csv pointing to an "
                            "existing ablation_results.csv (produced by --mode ablation).")
        ablation_df = pd.read_csv(args.ablation_csv)
        sig_df = run_statistical_tests(ablation_df, baseline_exp=args.baseline_exp,
                                       metric=args.tuning_metric)
        sig_df.to_csv(os.path.join(args.output_dir, "statistical_significance.csv"), index=False)
        logger.info("\n%s", sig_df.to_string())

    elif mode == "cross_dataset":
        if not args.cross_datasets:
            raise ValueError("--mode cross_dataset requires --cross_datasets ds1,ds2,...")
        run_cross_dataset_difficulty(args, args.cross_datasets, seeds=tuple(args.seeds[:1]))

    elif mode == "full":
        run_full_pipeline(args)

    else:
        raise ValueError(f"Unknown --mode {mode!r}.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="BiSLU + PABEE | Frequency-Adaptive Early Exit | HF datasets | "
                    "HPO + Ablation + Sensitivity + Statistical Testing + Visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", default="train",
                   choices=["train", "hpo", "final", "ablation", "sensitivity",
                           "stats", "cross_dataset", "full"],
                   help="Pipeline stage to run. 'train' is the original single-run behaviour.")
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
    p.add_argument("--train_batch_size",            default=8,   type=int)
    p.add_argument("--eval_batch_size",             default=4,    type=int)
    p.add_argument("--learning_rate",               default=1e-5, type=float)
    p.add_argument("--num_train_epochs",            default=12,   type=int)
    p.add_argument("--warmup_proportion",           default=0.1,  type=float)
    p.add_argument("--gradient_accumulation_steps", default=2,    type=int)
    p.add_argument("--weight_decay",                default=0.01, type=float)
    p.add_argument("--adam_epsilon",                default=1e-8, type=float)
    p.add_argument("--max_grad_norm",               default=1.0,  type=float)
    p.add_argument("--logging_steps",               default=200,  type=int)
    p.add_argument("--early_stopping",              default=5,    type=int)
    p.add_argument("--tuning_metric",               default="mean_intent_slot")
    p.add_argument("--seed", default=42, type=int,
                   help="Base seed for train/final runs and the sensitivity sweep.")
    p.add_argument("--loss_coef_intent",     default=0.5,  type=float)
    p.add_argument("--loss_coef_slot",       default=0.5,  type=float)
    p.add_argument("--loss_coef_slot_scl",   default=0.5,  type=float)
    p.add_argument("--loss_coef_intent_scl", default=0.5,  type=float)
    p.add_argument("--sd_loss_coef",         default=0.5,  type=float)
    p.add_argument("--use_soft_slot",                action="store_true")
    p.add_argument("--use_scl",                      action="store_true")
    p.add_argument("--use_sd",                       action="store_true")
    p.add_argument("--use_intent_context_attention", action="store_true")
    p.add_argument("--disable_early_exit", action="store_true",
                   help="Force full-depth inference (Ablation Experiment 1: standard "
                        "BiSLU, no PABEE early exit). Reuses the per-sample "
                        "min-exit-layer gate, so no separate forward path exists.")
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
    p.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=True,
                   help="Freeze the LLaMA backbone and train only the BiSLU+PABEE heads "
                        "(soft_intent, hard_intent, slot_clf, exit_intent_heads, "
                        "exit_slot_probes). This is now the default training regime; "
                        "pass --no-freeze_backbone to fully fine-tune the backbone as before.")

    # ------------------------------------------------------------------
    p.add_argument("--use_gc",  action=argparse.BooleanOptionalAction, default=True,
                   help="HF backbone gradient checkpointing. Provides a SMALLER benefit "
                        "than usual for this architecture: output_hidden_states=True is "
                        "required for the per-layer PABEE auxiliary losses regardless, so "
                        "all layer hidden states are materialized either way. Still saves "
                        "the intra-layer attention/MLP activations HF itself would "
                        "otherwise retain. See --use_layer_probe_checkpointing for the "
                        "fix that targets the larger, actually-avoidable cost.")
    p.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True,
                   help="bfloat16 autocast for forward/backward compute. Master weights "
                        "and AdamW optimizer state remain fp32.")
    p.add_argument("--use_layer_probe_checkpointing",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Exact activation checkpointing (Chen et al. 2016 — recomputes, "
                        "does not approximate, so gradients are unchanged) around the "
                        "per-layer auxiliary intent/slot probe heads in "
                        "JointModelWithEarlyExit.forward. This is the primary fix for the "
                        "O(L)-in-num-layers activation memory of PABEE-style joint "
                        "training with per-layer auxiliary losses; see the inline comment "
                        "at the checkpoint call site for the arithmetic. Training-only; "
                        "has no effect on forward_with_early_exit (already O(1) memory).")
    p.add_argument("--load_dtype", default="fp32", choices=["fp32", "bf16"],
                   help="Backbone weight storage dtype at AutoModel.from_pretrained load "
                        "time. Default fp32 (safe). 'bf16' roughly halves backbone weight "
                        "memory (~4GB -> ~2GB for a 1B model) but is an OPT-IN, not a "
                        "default, because full bf16 weight+optimizer training carries real "
                        "precision risk that this project has already hit once (bf16 NaN "
                        "in MLD self-distillation, per prior debugging). If you set this "
                        "to bf16, watch train/nonfinite_total closely.")
    p.add_argument("--max_consecutive_nonfinite", default=20, type=int,
                   help="Abort a run with an explicit RuntimeError after this many "
                        "CONSECUTIVE non-finite loss/gradient events (fail fast "
                        "instead of silently training a corrupted model for entire "
                        "epochs). HPO/ablation/sensitivity catch the RuntimeError "
                        "and record the configuration as failed.")
    p.add_argument("--use_wandb",        action="store_true")
    p.add_argument("--wandb_project",    default="bislu-pabee")
    p.add_argument("--wandb_entity",     default=None)
    p.add_argument("--wandb_run_name",   default=None)
    p.add_argument("--wandb_watch_freq", default=100, type=int)

    # ---- HPO (--mode hpo / final / full) -------------------------------
    p.add_argument("--run_hpo", type=lambda s: s.lower() != "false", default=True,
                   help="Whether 'final'/'full' modes should run HPO first "
                        "(set --run_hpo false to reuse CLI hyperparameters directly).")
    p.add_argument("--n_trials",      default=5, type=int, help="Optuna trial budget.")
    p.add_argument("--hpo_epochs",    default=8,  type=int, help="Epochs per HPO trial.")
    p.add_argument("--hpo_early_stopping", default=2, type=int)
    p.add_argument("--hpo_seed",      default=42, type=int)
    p.add_argument("--study_name",    default="bislu_pabee_hpo_v2",
                   help="Bumped to _v2 because the HPO suggest ranges changed; "
                        "mixing new trials into a study created with the old "
                        "(unstable) ranges would poison the TPE sampler with the "
                        "diverged high-LR trials. Delete the old optuna_study.db "
                        "or keep this new name.")
    p.add_argument("--hpo_subset_fraction", default=0.2, type=float,
               help="Fraction of training data to use for HPO trials (e.g., 0.2 for 20%)")

    # ---- Ablation / cross-dataset (--mode ablation / cross_dataset / full)
    p.add_argument("--seeds", default="42,43,44",
                   type=lambda s: [int(x) for x in s.split(",") if x.strip() != ""],
                   help="Comma-separated seed list for the ablation study / "
                        "cross-dataset sweep (n=3 is a minimal, still "
                        "underpowered, default — see statistical-testing caveat).")
    p.add_argument("--cross_datasets", default="",
                   type=lambda s: [x.strip() for x in s.split(",") if x.strip() != ""],
                   help="Comma-separated HF dataset names for --mode cross_dataset.")
    p.add_argument("--experiments", default="",
                   type=lambda s: [x.strip() for x in s.split(",") if x.strip() != ""],
                   help="Comma-separated list of ablation experiments to run")
    # ---- Statistical testing (--mode stats) -----------------------------
    p.add_argument("--ablation_csv", default="",
                   help="Path to an existing ablation_results.csv (for --mode stats).")
    p.add_argument("--baseline_exp", default="E1_baseline",
                   help="Ablation experiment name used as the significance-testing baseline.")

    # ---- Sensitivity analysis (--mode sensitivity) -----------------------
    p.add_argument("--sensitivity_epochs",         default=3, type=int)
    p.add_argument("--sensitivity_early_stopping", default=2, type=int)
    p.add_argument("--best_hp_json", default="",
                   help="best_hyperparameters.json from a prior HPO run, used to "
                        "center the --mode sensitivity OFAT sweep.")

    args = p.parse_args()
    if args.mode == "train" and not args.do_train and not args.do_eval:
        p.error("--mode train requires --do_train and/or --do_eval.")
    if args.use_scl and not args.use_soft_slot:
        p.error("--use_scl requires --use_soft_slot.")
    if not args.use_soft_slot:
        logger.warning("--use_soft_slot not set; soft-slot feedback disabled.")
    main(args)