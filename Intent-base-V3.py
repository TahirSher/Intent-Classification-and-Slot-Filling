import os, re, sys, json, math, inspect, warnings, logging, dataclasses, time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import trange

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
        f"_frozen_ee{args.ee_patience}_tau{args.tau_intent}"
        f"_minexit{args.min_exit_layer}"
        f"_freqexit{int(getattr(args,'use_freq_exit',False))}"
    )
    _wb.init(
        project=getattr(args, "wandb_project", "frozen-pabee-intent-slot"),
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
    """
    Builds:
      - intent_label_set : sorted intent strings + "UNK"
      - slot_label_set    : BIO TAG vocabulary, i.e. ["O", "B-<type>", "I-<type>", ...]
        (NOT raw entity types -- this is now a proper per-token tagging
        scheme decoded with the standard BIO chunk decoder, not a
        span-matrix scheme.)
    """
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
    slot_label_set = ["O"] + [f"{p}-{t}" for t in sorted(slot_type_set) for p in ("B", "I")]
    logger.info("Label sets: %d intents, %d BIO slot tags (%d entity types).",
                len(intent_label_set), len(slot_label_set), len(slot_type_set))
    return intent_label_set, slot_label_set


# ============================================================
# 2.5  WORD FREQUENCY INDEX
# ============================================================

class WordFrequencyIndex:
    """
    Maps each utterance to a rarity score in [0, 1] based on corpus word
    frequencies.  score=0 -> all words frequent -> safe to exit early.
    score=1 -> rare word present -> deep layers needed.
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
# 3.  BIO PARSING / CHUNK DECODING
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
# 4.  PRECISION / RECALL / F1  (span-set based, seqeval-equivalent)
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


def get_slot_label_lists(slot_label_ids, slot_logits, word_attention_mask, label_set):
    """
    Decode per-token BIO predictions (and gold labels) into entity-span sets
    using the standard BIO chunk decoder, then hand them to the seqeval-style
    P/R/F1 above. This replaces the old span-matrix decode logic entirely.
    """
    pred_ids = slot_logits.argmax(dim=-1)
    yt, yp = [], []
    for i in range(len(slot_label_ids)):
        tl = int(word_attention_mask[i].sum().item())
        gold_tags = [label_set[int(x)] for x in slot_label_ids[i][:tl].tolist()]
        pred_tags = [label_set[int(x)] for x in pred_ids[i][:tl].tolist()]
        yt.append(get_bio_entities(gold_tags))
        yp.append(get_bio_entities(pred_tags))
    return yt, yp


def compute_metrics(args, ip, il, sp, sl, wm, ls, intent_threshold: float = 0.5):
    """
    IMPORTANT METRIC CAVEAT (read before trusting "intent_acc" alone):
    for a multi-label utterance (MixATIS-style compound intents), "intent_acc"
    below is EXACT-MATCH SUBSET ACCURACY -- the entire predicted label set
    must match the gold set exactly, one wrong bit anywhere fails the whole
    sample. This is a legitimate, standard multi-label metric (Tsoumakas &
    Katakis, 2007), but it is punishing: a model that gets most labels right
    but misses one rare class on most samples can show near-zero subset
    accuracy while still being a reasonably good classifier. `intent_micro_f1`
    / `intent_macro_f1` below give the per-label picture and are usually the
    more informative numbers to optimize against for compound-intent tasks.
    """
    yt, yp = get_slot_label_lists(
        sl.detach().cpu(), sp.detach().float().cpu(), wm.detach().cpu(), ls)
    ip_float = ip.detach().float().cpu()
    il_cpu   = il.detach().cpu()
    single_intent = torch.all(il_cpu.sum(dim=1) == 1).item()
    if single_intent:
        pred_idx = ip_float.argmax(dim=1)
        gold_idx = il_cpu.argmax(dim=1)
        ipn = torch.zeros_like(il_cpu)
        ipn[torch.arange(il_cpu.size(0)), pred_idx] = 1
        ipn = ipn.numpy().astype(int); iln = il_cpu.numpy().astype(int)
    else:
        probs = torch.sigmoid(ip_float)
        ipn = (probs >= intent_threshold).numpy().astype(int)
        iln = il_cpu.numpy().astype(int)

    ia = accuracy_score(iln, ipn)  # STRICT exact-match subset accuracy -- see caveat above
    p_mi, r_mi, f_mi, _ = precision_recall_fscore_support(
        iln, ipn, average="micro", zero_division=0)
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(
        iln, ipn, average="macro", zero_division=0)

    sfa = float(np.mean(
        np.all(ipn == iln, axis=1) &
        np.array([set(map(tuple,p))==set(map(tuple,t)) for p,t in zip(yp,yt)])
    ))
    f = seq_f1(yt, yp)
    return {
        "intent_acc":            ia,     # kept for backward compat -- STRICT, see caveat
        "intent_micro_f1":       f_mi,   # per-label micro F1 (recommended primary intent metric)
        "intent_micro_precision":p_mi,
        "intent_micro_recall":   r_mi,
        "intent_macro_f1":       f_ma,   # unweighted across classes -- sensitive to rare-class misses
        "intent_threshold_used": intent_threshold,
        "slot_precision":   seq_prec(yt, yp),
        "slot_recall":      seq_rec(yt, yp),
        "slot_f1":          f,
        "mean_intent_slot": (ia + f) / 2.0,   # unchanged formula, kept for backward compat -- STRICT
        "mean_f1":          (f_mi + f) / 2.0, # NEW: less punishing composite (micro-F1 based)
        "semantic_acc":     sfa,
    }


def search_best_intent_threshold(probs: np.ndarray, labels: np.ndarray,
                                 grid: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Grid search over the sigmoid decision threshold to maximize micro-F1 on
    the given (probs, labels) pair. Standard decision-threshold calibration
    for multi-label classification (see e.g. Zhang & Zhou, 2013 survey of
    multi-label learning, section on thresholding strategies). Intended to
    be called on the DEV split only, then frozen and reused on test to avoid
    threshold leakage.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        preds = (probs >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


# ============================================================
# 5.  MASK UTILITY
# ============================================================

def get_useful_ones(out, label, mask):
    """
    Flattens (B, T, C) logits / (B, T) labels / (B, T) mask down to only the
    valid (non-padding) token positions. Works uniformly for per-token slot
    classification at any layer.
    """
    fm = mask.reshape(-1).bool()
    fo = out.reshape(-1, out.shape[-1])
    fl = label.reshape(-1)
    idx = fm.nonzero(as_tuple=False).squeeze(-1).long()
    return fo.index_select(0, idx), fl.index_select(0, idx)


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
        self.o_id            = self.slot_label_id.get("O", 0)
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

    def _bio_label_seq(self, entities):
        """
        Word-position BIO tag-id sequence, length self.max_seq.
        Offset convention (+1 for BOS pseudo-word) matches `_tokenise` above
        and is preserved unchanged from the original script -- see the
        CORRECTNESS NOTE at the top of this file regarding tokenizers
        without a BOS token.
        """
        labels = torch.full((self.max_seq,), self.o_id, dtype=torch.long)
        for etype, es, ee in entities:
            si, ei = es + 1, ee + 1
            if si >= self.max_seq:
                continue
            ei = min(ei, self.max_seq - 1)
            b_id = self.slot_label_id.get(f"B-{etype}")
            i_id = self.slot_label_id.get(f"I-{etype}")
            if b_id is None:
                continue
            labels[si] = b_id
            if i_id is not None and ei > si:
                labels[si + 1: ei + 1] = i_id
        return labels

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
            slot_lbl = self._bio_label_seq(entities)
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
                slot_lbl = self._bio_label_seq(ents)
            else:
                slot_lbl = torch.full((self.max_seq,), self.o_id, dtype=torch.long)
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

def asymmetric_loss(y_hat: torch.Tensor, y_true: torch.Tensor,
                    gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                    clip: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    """
    Asymmetric Loss (ASL) for multi-label classification
    (Ben-Baruch et al., 2020, "Asymmetric Loss For Multi-Label Classification").

    NOVELTY vs the plain BCE+pos_weight this can replace as intent loss: ASL
    attacks the same class-imbalance problem pos_weight does (rare intents
    contributing tiny gradient), but with two mechanisms pos_weight lacks:
      1. Asymmetric focal focusing -- easy negatives (gamma_neg, large) are
         down-weighted far more aggressively than positives (gamma_pos,
         usually 0), so the huge majority of easy-negative intent classes
         per sample stop dominating the gradient, without needing a
         hand-tuned per-class pos_weight table that has to be clipped
         (max_weight=50 above) to avoid blowing up on singleton classes.
      2. Probability shifting ("clip"): negative probabilities are nudged
         up by `clip` before the loss/focusing term is computed, hard-
         discarding very-easy negatives rather than merely down-weighting
         them. Useful for compound-intent, multi-label targets like
         MixATIS where the negative:positive ratio per class can be >>50:1
         -- a regime plain BCE and even clipped pos_weight both struggle
         with.
    """
    y_true = y_true.float()
    p = torch.sigmoid(y_hat.float())
    p_pos = p
    p_neg = 1.0 - p
    if clip is not None and clip > 0:
        p_neg = (p_neg + clip).clamp(max=1.0)

    los_pos = y_true * torch.log(p_pos.clamp(min=eps))
    los_neg = (1.0 - y_true) * torch.log(p_neg.clamp(min=eps))
    loss = los_pos + los_neg

    if gamma_neg > 0 or gamma_pos > 0:
        pt0 = p_pos * y_true
        pt1 = p_neg * (1.0 - y_true)
        pt = pt0 + pt1
        gamma = gamma_pos * y_true + gamma_neg * (1.0 - y_true)
        focal_weight = torch.pow((1.0 - pt).clamp(min=0.0), gamma)
        loss = loss * focal_weight

    return -loss.mean()


def intent_loss_func(y_hat, y_true, pos_weight: Optional[torch.Tensor] = None,
                     loss_fn: str = "bce", asl_gamma_neg: float = 4.0,
                     asl_gamma_pos: float = 0.0, asl_clip: float = 0.05):
    if loss_fn == "asl":
        return asymmetric_loss(y_hat, y_true, gamma_neg=asl_gamma_neg,
                               gamma_pos=asl_gamma_pos, clip=asl_clip)
    return F.binary_cross_entropy_with_logits(
        y_hat.float(), y_true.float(), pos_weight=pos_weight)


def compute_intent_pos_weight(hf_train, int_field, is_instruction,
                              intent_label_set, max_weight: float = 50.0) -> torch.Tensor:
    """
    Per-class positive weight for BCEWithLogitsLoss, i.e. (n_negative / n_positive)
    per intent class, clipped to [0.1, max_weight]. This is the standard PyTorch-
    documented correction for class imbalance in multi-label BCE (see
    torch.nn.BCEWithLogitsLoss docs, "pos_weight"; also King & Zeng, 2001, on
    rare-event correction in binary classification more generally).

    Without this, a class that's positive in e.g. 2% of utterances contributes
    ~50x less positive-direction gradient than negative-direction gradient per
    epoch, so the model can trivially minimize BCE by pushing that class's
    logit very negative and never firing on it -- exactly the pattern of
    "loss keeps falling, subset accuracy stays near the multi-label chance
    floor" you're seeing.
    """
    n = len(intent_label_set)
    counts = np.zeros(n, dtype=np.float64)
    total = 0
    idx_of = {w: i for i, w in enumerate(intent_label_set)}
    for row in hf_train:
        total += 1
        if is_instruction:
            intents = parse_intents(row["completion"]).split('#')
        else:
            raw_int = row[int_field]
            intents = ([str(x) for x in raw_int] if isinstance(raw_int, list)
                      else str(raw_int).replace(',', '#').split('#'))
        for it in intents:
            it = it.strip()
            if not it:
                continue
            j = idx_of.get(it, idx_of.get("UNK"))
            if j is not None:
                counts[j] += 1
    pos = np.clip(counts, 1.0, None)
    neg = np.clip(total - counts, 0.0, None)
    pw  = np.clip(neg / pos, 0.1, max_weight)
    logger.info(
        "Intent pos_weight (class-imbalance correction): min=%.2f max=%.2f mean=%.2f "
        "median=%.2f (clipped to [0.1, %.1f]); %d/%d classes hit the upper clip.",
        pw.min(), pw.max(), pw.mean(), float(np.median(pw)), max_weight,
        int((pw >= max_weight).sum()), n,
    )
    return torch.tensor(pw, dtype=torch.float32)


# ============================================================
# 8.5  SUPERVISED CONTRASTIVE LOSS (SCL)
# ============================================================
#
# Design summary (read before touching --use_scl):
#
# The backbone is frozen and run in eval()/no_grad() (see
# JointModelWithEarlyExit.forward), so for a fixed input, the per-layer
# pooled features (cls_l, word_h_l) are exactly deterministic -- there is no
# backbone-side stochasticity available to build "two augmented views" of a
# sample the way image-domain SupCon (Khosla et al., 2020) does with random
# crops. Recomputing the backbone a second time under a different dropout
# rate (as some frozen-backbone SCL implementations do) is also not an
# option here without literally unfreezing/perturbing the backbone, which
# would defeat the point of freezing it.
#
# Instead, positive pairs are built the SimCSE way (Gao et al., 2021,
# "SimCSE: Simple Contrastive Learning of Sentence Embeddings"): the SAME
# frozen feature is passed twice through a small trainable projection head
# (ContrastiveProjectionHead, below) that has its own internal dropout.
# Two independent dropout masks -> two slightly different embeddings of the
# same underlying representation -- "dropout as minimal data augmentation".
# This is a genuine adaptation, not a literal instantiation of either paper:
# SimCSE's original recipe is unsupervised (in-batch negatives, no label
# information); what's implemented below is the SimCSE view-construction
# trick combined with a SUPERVISED (label-aware) positive/negative
# assignment, i.e. the positive set for an anchor is not just its own other
# dropout view but every same-label sample's views too, as in Khosla et al.
# Flagging this explicitly since it is a hybrid of two different papers'
# ideas, not a direct reproduction of either.
#
# Two separate contrastive objectives are provided:
#
#   scl_intent_loss  -- utterance-level, multi-label aware. Standard SupCon
#     (Khosla et al. 2020, Eq. 2) is only defined for single-label targets
#     (positive = "same one-hot class"). Intent labels here are multi-hot
#     (compound intents), so the binary "same class" positive indicator is
#     replaced by a continuous Jaccard-similarity weight between two
#     samples' multi-hot label sets. This is a mathematically consistent
#     generalization -- for one-hot labels, Jaccard(same class) = 1 and
#     Jaccard(different classes) = 0, which reduces EXACTLY to Khosla et
#     al.'s binary mask -- but it is a generalization, not the literal
#     published formula, since Khosla et al. never define a multi-label
#     positive weighting scheme. Said plainly: this is a reasonable,
#     defensible extension, not a citation of an established result for
#     the multi-label case.
#
#   scl_slot_loss  -- token-level, single-label (each token has exactly one
#     BIO tag), so this IS the literal Khosla et al. 2020 SupCon formula
#     (binary same-tag positive mask), applied per valid word position
#     instead of per utterance. See the KNOWN LIMITATION note in its
#     docstring regarding the 'O'-tag class imbalance -- that is a real,
#     un-addressed weakness of applying vanilla SupCon to BIO tagging as-is.

def _weighted_supcon_loss(sim: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Numerically-stable weighted supervised contrastive loss, "Lout" formulation
    (Khosla et al., 2020, Eq. 2: normalize by the number/weight of positives
    per anchor, average the resulting per-anchor loss over anchors that have
    at least one positive).

    `weight[i, j] in [0, 1]` is the "how positive" score between anchor i and
    candidate j (self-pairs excluded via `eye`). `weight` binary {0, 1}
    recovers the exact literature formula; `weight` continuous (Jaccard,
    used by scl_intent_loss below) is the generalization documented in the
    module docstring above.

    sim must already be the raw (unmasked, un-normalized-by-logsumexp)
    cosine-similarity-over-temperature matrix; masking of the diagonal and
    the log-sum-exp denominator are both handled here so every caller gets
    identical numerical treatment.
    """
    eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    sim = sim.float()
    weight = weight.float().masked_fill(eye, 0.0)
    sim = sim.masked_fill(eye, -1e9)

    log_den  = torch.logsumexp(sim, dim=1, keepdim=True)
    log_prob = sim - log_den

    weight_sum = weight.sum(dim=1)
    valid = weight_sum > 1e-8
    if valid.sum() == 0:
        return sim.sum() * 0.0

    loss_per_row = -(log_prob * weight).sum(dim=1) / weight_sum.clamp(min=1e-8)
    return loss_per_row[valid].mean()


def scl_intent_loss(views: torch.Tensor, intent_labels: torch.Tensor,
                    temp: float = 0.10) -> torch.Tensor:
    """
    views:         (B, V, P) L2-normalized projection-head embeddings.
                   V = number of dropout views per utterance (2 in this
                   file, see ContrastiveProjectionHead / _layer_losses).
    intent_labels: (B, C) multi-hot ground-truth intent vector.

    Positive weight between (view of sample i) and (view of sample j) is the
    Jaccard similarity of their multi-hot label sets -- see module docstring
    for why this, rather than a binary "any shared label" threshold, is the
    more principled choice: it does not treat "shares 1 of 5 active labels"
    identically to "shares 5 of 5 active labels".
    """
    B, V, P = views.shape
    N = B * V
    flat = views.reshape(N, P)
    sim  = flat @ flat.T / temp

    lbl    = intent_labels.float()
    inter  = lbl @ lbl.T
    card   = lbl.sum(dim=-1, keepdim=True)
    union  = card + card.T - inter
    jacc   = inter / union.clamp(min=1e-8)
    jacc   = jacc.repeat_interleave(V, dim=0).repeat_interleave(V, dim=1)

    return _weighted_supcon_loss(sim, jacc)


def scl_slot_loss(views: torch.Tensor, slot_labels: torch.Tensor,
                  word_attention_mask: torch.Tensor, temp: float = 0.10,
                  max_tokens: int = 512) -> torch.Tensor:
    """
    views:               (B, MW, V, P) L2-normalized per-word-position
                         projection-head embeddings, V dropout views/token.
    slot_labels:         (B, MW) gold BIO tag ids.
    word_attention_mask: (B, MW) 1 for real words, 0 for padding.

    Positive weight = exact BIO tag equality (single-label per token, so
    this is literally Khosla et al. 2020's binary-mask SupCon -- no
    generalization needed here, unlike the intent loss above).

    KNOWN LIMITATION (stated plainly, not glossed over): BIO tagging is
    dominated by the 'O' class. This implementation does NOT down-weight or
    subsample 'O' positives -- every 'O' token is a positive for every other
    'O' token, batch-wide. In a batch where most tokens are 'O', that
    majority-class block will dominate both the positive set and the
    denominator for most anchors, diluting the gradient signal for the
    rarer, more decision-relevant B-/I- tokens. If slot metrics do not
    improve (or regress) after enabling --use_scl, this is the first thing
    to investigate -- e.g. capping the number of 'O' tokens sampled per
    batch -- not something the loss below corrects for on its own.

    `max_tokens` randomly subsamples valid word positions before building
    the similarity matrix, purely to bound the O(N^2) cost of the
    similarity matrix for long-sequence batches; it does not address the
    'O'-class imbalance above (the subsample is uniform, not class-balanced).
    """
    B, MW, V, P = views.shape
    mask = word_attention_mask.bool()
    valid_idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    n_valid = valid_idx.numel()
    if n_valid == 0:
        return views.sum() * 0.0

    flat_views  = views.reshape(B * MW, V, P).index_select(0, valid_idx)   # (n_valid, V, P)
    flat_labels = slot_labels.reshape(-1).index_select(0, valid_idx)        # (n_valid,)

    if n_valid > max_tokens:
        perm = torch.randperm(n_valid, device=views.device)[:max_tokens]
        flat_views  = flat_views.index_select(0, perm)
        flat_labels = flat_labels.index_select(0, perm)
        n_valid = max_tokens

    N = n_valid * V
    flat = flat_views.reshape(N, P)
    sim  = flat @ flat.T / temp

    pos = (flat_labels.unsqueeze(0) == flat_labels.unsqueeze(1))
    pos = pos.repeat_interleave(V, dim=0).repeat_interleave(V, dim=1).float()

    return _weighted_supcon_loss(sim, pos)


# ============================================================
# 9.  PER-LAYER CLASSIFICATION HEAD
# ============================================================

class ExitHead(nn.Module):
    """
    Per-layer probe reused for both the intent head and the slot head (with
    different output dims and, optionally, different capacity).

    Two additions versus a bare Linear probe, both standard in the linear/
    non-linear probing literature:

    1. LayerNorm BEFORE the classifier. Raw hidden-state magnitude typically
       drifts with depth in transformer stacks; without normalization a
       probe trained on layer 8's activation scale may be poorly conditioned
       relative to layer 15's. Tenney et al. (2019, "BERT Rediscovers the
       Classical NLP Pipeline") normalize pooled representations before
       their per-layer probes for exactly this reason.
    2. Optional single hidden layer (`hidden_dim > 0`). A pooled, single-
       vector-per-utterance intent classifier over a COMBINATORIAL,
       multi-label target space is a harder decision boundary than per-
       token BIO tagging (which gets one gradient signal per token, is
       lexically local, and only needs to separate O/B/I within a small
       local context). Giving the intent head modest non-linear capacity is
       a defensible, minimal departure from strict linear probing (see
       Belinkov, 2022, "Probing Classifiers" survey, on the linear-vs-MLP
       probe capacity tradeoff) -- it costs a few hundred KB of parameters
       and is disabled (hidden_dim=0 -> pure linear) by default for the
       slot head, which already performs well as a linear probe.
    """
    def __init__(self, hidden_size: int, num_labels: int,
                dropout_rate: float = 0.1, hidden_dim: int = 0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_labels),
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, num_labels),
            )

    def forward(self, x):
        return self.net(self.norm(x))


class ContrastiveProjectionHead(nn.Module):
    """
    Small MLP "projection network" (Khosla et al., 2020, Sec. 3: SupCon maps
    the encoder representation r through a projection network to a
    normalized embedding z, and the contrastive loss is computed on z, not
    on r or on classification logits). Deliberately a SEPARATE module from
    ExitHead rather than a repurposed classifier output, for two reasons:

    1. Applying a contrastive loss directly to classification logits (or to
       whatever ExitHead's Linear layer produces) couples two objectives
       with different geometric goals -- softmax/BCE wants class-separating
       margins in a `num_labels`-dimensional space, SupCon wants
       label-clustered directions on a unit hypersphere in a (typically
       different-dimensional) embedding space. Sharing one output layer
       between them fights both objectives.
    2. Discarding the projection head at inference (only ExitHead's output
       is ever used by forward_with_early_exit) matches the SupCon paper's
       own recommended usage: the projection network is auxiliary scaffolding
       for shaping the representation during training, not part of the
       deployed model.

    Own LayerNorm + own Dropout (independent from ExitHead's) is required
    for the SimCSE-style dropout-based view construction used by
    scl_intent_loss / scl_slot_loss -- see module-level SCL docstring above.
    """
    def __init__(self, hidden_size: int, proj_dim: int = 128, dropout_rate: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.net = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(self.norm(x))
        return F.normalize(z.float(), dim=-1)


class LastTokenPooling(nn.Module):
    """
    Pool the last non-padding token's hidden state.

    Uses `attn_mask.shape[0]` (always the batch dim) rather than
    `h.shape[0]`, which under certain HF/SDPA paths can arrive as the
    sequence dim instead -- see original bug note preserved from the prior
    version of this script.
    """
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
        self.base_model = AutoModel.from_pretrained(args.model_name_or_path)
        self.pooling    = LastTokenPooling()

    def forward(self, input_ids, attention_mask, words_lengths):
        with torch.no_grad():
            out = self.base_model(input_ids, attention_mask=attention_mask,
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
    Build the kwargs dict for a single decoder layer call, using
    `inspect.signature` so this works across transformers versions that
    renamed `past_key_value` -> `past_key_values` or added/removed
    `cache_position`.
    """
    sig    = inspect.signature(layer.forward)
    params = sig.parameters
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    candidates: Dict[str, object] = {
        "attention_mask":    causal_mask,
        "position_ids":      position_ids,
        "past_key_value":    None,
        "past_key_values":   None,
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
# 10. FROZEN-BACKBONE JOINT MODEL WITH FREQUENCY-ADAPTIVE PABEE EXIT
# ============================================================

class JointModelWithEarlyExit(nn.Module):
    def __init__(self, args, num_intent, num_slot):
        super().__init__()
        self.args          = args
        self.num_intent    = num_intent
        self.num_slot      = num_slot
        self.use_freq_exit = getattr(args, "use_freq_exit", False)
        self.use_scl       = getattr(args, "use_scl", False)

        cfg             = AutoConfig.from_pretrained(args.model_name_or_path)
        self.num_layers = cfg.num_hidden_layers
        self.wordrep    = DecoderWordRep(args)

        # ---- Freeze the entire backbone. Only the per-layer heads train. ----
        for p in self.wordrep.base_model.parameters():
            p.requires_grad_(False)
        self.wordrep.base_model.eval()

        intent_hidden = getattr(args, "intent_head_hidden", 0)
        slot_hidden   = getattr(args, "slot_head_hidden", 0)
        self.exit_intent_heads = nn.ModuleList([
            ExitHead(cfg.hidden_size, num_intent, args.dropout_rate, hidden_dim=intent_hidden)
            for _ in range(self.num_layers)])
        self.exit_slot_heads = nn.ModuleList([
            ExitHead(cfg.hidden_size, num_slot, args.dropout_rate, hidden_dim=slot_hidden)
            for _ in range(self.num_layers)])

        # ---- SCL projection heads (only built when --use_scl is set). ----
        # Separate ModuleList per layer per task, mirroring exit_intent_heads /
        # exit_slot_heads -- one small projection network per depth, since a
        # single shared projection across all layers would force early and
        # late layers (very different representational content, per Tenney
        # et al. 2019) into the same contrastive embedding space.
        if self.use_scl:
            proj_dim     = getattr(args, "scl_proj_dim", 128)
            scl_dropout  = getattr(args, "scl_dropout_rate", 0.1)
            self.scl_intent_proj = nn.ModuleList([
                ContrastiveProjectionHead(cfg.hidden_size, proj_dim, scl_dropout)
                for _ in range(self.num_layers)])
            self.scl_slot_proj = nn.ModuleList([
                ContrastiveProjectionHead(cfg.hidden_size, proj_dim, scl_dropout)
                for _ in range(self.num_layers)])

        _bdt = next(self.wordrep.base_model.parameters()).dtype
        for _cn, _cm in self.named_children():
            if _cn != "wordrep":
                _cm.to(dtype=_bdt)

        # ---- Enforce the >= 50%-depth exit floor. ----
        half = math.ceil(self.num_layers / 2)
        raw_mel = getattr(args, "min_exit_layer", None)
        if raw_mel is None:
            self.min_exit_layer = half
        elif raw_mel < half:
            logger.warning(
                "--min_exit_layer=%d is below 50%% of depth (num_layers=%d -> "
                "floor=%d). Clamping to %d: this architecture never exits "
                "before the halfway point.", raw_mel, self.num_layers, half, half,
            )
            self.min_exit_layer = half
        else:
            self.min_exit_layer = raw_mel

        self.patience     = getattr(args, "ee_patience", 3)
        self.tau_intent   = getattr(args, "tau_intent",  0.05)  # kept for CLI back-compat; unused by the
                                                                 # corrected exit criterion below (see
                                                                 # forward_with_early_exit docstring)
        self.tau_slot     = getattr(args, "tau_slot",    0.1)
        self.intent_margin = getattr(args, "intent_exit_margin", 0.15)

        # ---- NOVELTY: depth-adaptive PABEE patience. ----
        # layer_savings_pct was only ~7% and pct_full_pass ~49% under a flat
        # patience -- a constant patience treats agreement at layer
        # min_exit_layer+1 (right at the 50%-depth floor, least trustworthy)
        # identically to agreement deep in the stack (most trustworthy),
        # which is overly conservative once you're well past the floor.
        # required_patience(l) = max(patience_min, patience - decay*(l - per_min))
        # decays the number of consecutive stable layers needed as depth
        # increases past the per-sample minimum, while the existing
        # confidence-margin + joint intent/slot agreement gate (unchanged)
        # still guards against a lucky-but-wrong early agreement. decay=0
        # (default off) reproduces the exact original flat-patience
        # behaviour.
        self.patience_decay = getattr(args, "ee_patience_decay", 0.0)
        self.patience_min   = getattr(args, "ee_patience_min", 1)

        # ---- NOVELTY: free 2-layer logit smoothing at the exit point. ----
        # The layer immediately before the one that triggers exit has
        # already been computed (it's how "stable" was judged) -- averaging
        # its logits with the exiting layer's costs zero extra FLOPs and
        # reduces single-layer-head noise/variance in the final prediction,
        # the same intuition as a 2-model logit ensemble.
        self.exit_logit_smoothing = getattr(args, "exit_logit_smoothing", True)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        logger.info(
            "Model: L=%d  min_exit=%d (floor=%d)  patience=%d (decay=%.2f min=%d)  "
            "tau_intent=%.4f  tau_slot=%.4f  freq_adaptive_exit=%s  "
            "exit_logit_smoothing=%s  use_scl=%s  trainable_params=%d/%d (%.2f%%)",
            self.num_layers, self.min_exit_layer, half, self.patience,
            self.patience_decay, self.patience_min,
            self.tau_intent, self.tau_slot, self.use_freq_exit,
            self.exit_logit_smoothing, self.use_scl,
            n_trainable, n_total, 100.0 * n_trainable / max(n_total, 1),
        )

    def train(self, mode: bool = True):
        """Backbone stays frozen and in eval() regardless of trainer state."""
        super().train(mode)
        self.wordrep.base_model.eval()
        return self

    # ------------------------------------------------------------------
    def forward(self, input_ids, attention_mask, words_lengths,
                word_attention_mask, return_layer_probes=False):
        """
        return_layer_probes=False (default, used by forward_with_early_exit's
        callers that only need the final layer): returns (final_intent_logits,
        final_slot_logits).

        return_layer_probes=True (used by EarlyExitTrainer.compute_loss):
        returns (l_int, l_slot, l_cls, l_word_h) -- per-layer classification
        logits AND the raw pooled features each layer's heads consumed. The
        raw features are exposed so the trainer can additionally route them
        through the (optional) SCL projection heads without recomputing the
        frozen backbone a second time -- see EarlyExitTrainer._layer_losses.
        """
        device = input_ids.device
        with torch.no_grad():
            out = self.wordrep.base_model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True,
            )
        hs    = out.hidden_states  # tuple: len = num_layers + 1 (embeddings + each block)
        align = _build_align(input_ids, words_lengths, device).to(dtype=hs[-1].dtype)

        l_int, l_slot, l_cls, l_word_h = [], [], [], []
        for l in range(self.num_layers):
            h        = hs[l + 1]
            cls_l    = self.wordrep.pooling(h, attention_mask)
            word_h_l = torch.bmm(align, h)
            l_int.append(self.exit_intent_heads[l](cls_l))
            l_slot.append(self.exit_slot_heads[l](word_h_l))
            if return_layer_probes:
                l_cls.append(cls_l)
                l_word_h.append(word_h_l)

        if not return_layer_probes:
            return l_int[-1], l_slot[-1]
        return l_int, l_slot, l_cls, l_word_h

    # ------------------------------------------------------------------
    def _true_layer_iter(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Generator[Tuple[int, torch.Tensor], None, None]:
        """
        Layer-by-layer generator.  Yields (layer_idx, hidden_states) after
        each transformer block.  Breaking the outer loop stops computation
        at that layer -- remaining blocks are never executed (true FLOPs
        saving). The backbone is frozen, so no autograd bookkeeping is
        needed here at all; the whole generator runs under the caller's
        `torch.no_grad()`.
        """
        bm     = self.wordrep.base_model
        device = input_ids.device
        B, T   = input_ids.shape

        if not (hasattr(bm, 'embed_tokens') and hasattr(bm, 'layers')):
            raise RuntimeError(
                "Backbone does not expose .embed_tokens / .layers. "
                "True layer-by-layer early exit is unsupported for this model."
            )

        h = bm.embed_tokens(input_ids)

        position_ids   = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        cache_position = torch.arange(T, device=device)

        if hasattr(bm, 'rotary_emb'):
            position_embeddings = bm.rotary_emb(h, position_ids)
        else:
            position_embeddings = None

        causal_mask: Optional[torch.Tensor] = None
        if hasattr(bm, '_update_causal_mask'):
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

        for l, layer in enumerate(bm.layers):
            kw = _layer_kwargs_for(
                layer, causal_mask, position_ids, cache_position, position_embeddings
            )
            layer_out = layer(h, **kw)

            if isinstance(layer_out, torch.Tensor):
                raw = layer_out
            elif isinstance(layer_out, (tuple, list)):
                raw = layer_out[0]
            else:
                raw = layer_out[0]

            if raw.dim() != 3:
                raise RuntimeError(
                    f"Layer {l} output has unexpected shape {raw.shape}; "
                    f"expected 3-D (B={B}, T={T}, d)."
                )
            if raw.shape[0] == T and raw.shape[1] == B and T != B:
                logger.warning(
                    "Layer %d output appears transposed (%s); "
                    "correcting to (B, T, d).", l, tuple(raw.shape)
                )
                raw = raw.transpose(0, 1).contiguous()

            h = raw
            yield l, h

    # ------------------------------------------------------------------
    @staticmethod
    def _discretize_intent(ip_l: torch.Tensor, is_multi_label: bool,
                           intent_threshold: float) -> torch.Tensor:
        """
        Turns raw sigmoid probabilities into the discrete label set that
        `compute_metrics` would actually score, so the exit criterion is
        evaluated in the same space as accuracy rather than in raw
        probability space. For single-label utterances this is the standard
        argmax one-hot; for multi-label (compound-intent) utterances this is
        the thresholded bit-vector used everywhere else in this file.
        """
        if is_multi_label:
            return ip_l >= intent_threshold
        idx = ip_l.argmax(dim=-1)
        onehot = torch.zeros_like(ip_l, dtype=torch.bool)
        onehot.scatter_(1, idx.unsqueeze(1), True)
        return onehot

    @torch.no_grad()
    def forward_with_early_exit(
        self,
        input_ids:           torch.Tensor,
        attention_mask:      torch.Tensor,
        words_lengths:       torch.Tensor,
        word_attention_mask: torch.Tensor,
        freq_scores: Optional[torch.Tensor] = None,
        intent_threshold: float = 0.5,
        is_multi_label: bool = True,
    ) -> Tuple:
        """
        True layer-by-layer inference with:
          1. a per-sample MINIMUM exit layer interpolated from lexical
             rarity between `min_exit_layer` (>= 50% depth, enforced in
             __init__) and `num_layers - 1`;
          2. PABEE patience counting beyond that minimum, requiring BOTH
             the discretized intent label set AND the slot tag sequence to
             be stable relative to the previous layer before the patience
             counter increments (joint agreement -- either signal moving
             resets the counter). Intent stability is judged on the
             thresholded label set plus a confidence margin, not on raw
             sigmoid magnitude drift -- see the inline comment where
             `intent_agree` / `margin_ok` are computed for why the earlier
             magnitude-delta version could lock onto a stable-but-wrong
             ("predict nothing") prediction;
          3. the EXIT LAYER'S OWN head outputs as the final prediction
             (no separate reconstruction head -- see module docstring).

        FLOPs ~ (max_exit_layer_in_batch + 1) / L of a full forward pass;
        the `break` fires when the last unexited sample in the batch exits.

        NOTE: SCL projection heads (self.scl_intent_proj / self.scl_slot_proj)
        are intentionally NOT used anywhere in this method. They exist purely
        to shape exit_intent_heads / exit_slot_heads' input representations
        during training (see EarlyExitTrainer._layer_losses); at inference the
        exit heads' own outputs are used directly, exactly as before SCL was
        added -- this method's control flow and outputs are unchanged by
        --use_scl.
        """
        B      = input_ids.size(0)
        device = input_ids.device
        dummy  = next(self.wordrep.base_model.parameters())
        align  = _build_align(input_ids, words_lengths, device).to(dtype=dummy.dtype)
        wam_f  = word_attention_mask.float()
        wam_len = wam_f.sum(dim=1).clamp(min=1.0)

        if self.use_freq_exit and freq_scores is not None:
            fs      = freq_scores.float().to(device).clamp(0.0, 1.0)
            span    = float(self.num_layers - 1 - self.min_exit_layer)
            per_min = (self.min_exit_layer + fs * span).long().clamp(
                self.min_exit_layer, self.num_layers - 1
            )
        else:
            per_min = torch.full((B,), self.min_exit_layer,
                                 dtype=torch.long, device=device)

        pat_cnt  = torch.zeros(B, dtype=torch.long,  device=device)
        exited   = torch.zeros(B, dtype=torch.bool,  device=device)
        exit_lyr = torch.full((B,), self.num_layers - 1,
                              dtype=torch.long, device=device)
        exit_int:  List[Optional[torch.Tensor]] = [None] * B
        exit_slot: List[Optional[torch.Tensor]] = [None] * B

        prev_ip:           Optional[torch.Tensor] = None
        prev_intent_pred:  Optional[torch.Tensor] = None
        prev_slot_pred:    Optional[torch.Tensor] = None
        prev_int_logits:   Optional[torch.Tensor] = None
        prev_slot_logits:  Optional[torch.Tensor] = None
        last_int_logits:  Optional[torch.Tensor] = None
        last_slot_logits: Optional[torch.Tensor] = None

        for l, h in self._true_layer_iter(input_ids, attention_mask):
            cls_l    = self.wordrep.pooling(h, attention_mask)
            word_h_l = torch.bmm(align, h)

            int_logits_l  = self.exit_intent_heads[l](cls_l)
            slot_logits_l = self.exit_slot_heads[l](word_h_l)
            last_int_logits, last_slot_logits = int_logits_l, slot_logits_l

            ip_l          = torch.sigmoid(int_logits_l)
            slot_pred_l   = slot_logits_l.argmax(dim=-1)
            intent_pred_l = self._discretize_intent(ip_l, is_multi_label, intent_threshold)

            if prev_intent_pred is not None:
                eligible = (~exited) & (l >= per_min)

                # --- CORRECTNESS-BASED (true PABEE) patience criterion ---
                # Zhou et al., 2020 ("BERT Loses Patience") count patience on
                # agreement of the *discretized* prediction across consecutive
                # layers, not on raw magnitude drift of the pre-threshold
                # probabilities. A head that outputs uniformly near-zero
                # sigmoid probabilities ("predict nothing") barely changes in
                # absolute terms from layer to layer, so a magnitude-delta
                # criterion (the previous version of this method) can label
                # that degenerate, wrong output "stable" and lock the model
                # into it at the first eligible layer -- silently reproducing
                # "loss keeps falling, dev accuracy stays low" even though the
                # loss curve looks fine. Agreement of the thresholded label
                # set does not have this failure mode: an all-negative
                # prediction only "agrees" with another all-negative
                # prediction, it never averages away discriminative signal.
                intent_agree = (intent_pred_l == prev_intent_pred).all(dim=-1)

                # Confidence gate (patience combined with a minimum-margin
                # requirement, as in confidence-gated early-exit transformer
                # variants): agreement is only trusted once at least one
                # intent logit has moved outside the [-margin, +margin] band
                # around the decision threshold. This blocks the collapse
                # mode above -- a head sitting exactly at the threshold with
                # flat probabilities "agreeing with itself" every layer --
                # without touching the slot criterion, which was already
                # discrete/argmax-based and did not have this bug.
                margin_ok     = ((ip_l - intent_threshold).abs().amax(dim=-1) >= self.intent_margin)
                intent_stable = intent_agree & margin_ok

                disagree      = (slot_pred_l != prev_slot_pred).float() * wam_f
                frac_disagree = disagree.sum(dim=1) / wam_len
                slot_stable   = frac_disagree <= self.tau_slot

                joint_stable = intent_stable & slot_stable
                stable       = eligible & joint_stable
                unstable     = eligible & (~joint_stable)
                pat_cnt      = (pat_cnt + stable.long()) * (~unstable).long()

                # Depth-adaptive required patience (see __init__ docstring):
                # shrinks linearly with depth past each sample's own
                # per-sample floor, never below patience_min. decay=0
                # reproduces the original flat `self.patience` exactly.
                depth_since_min  = (l - per_min).clamp(min=0).float()
                required_patience = (self.patience - self.patience_decay * depth_since_min
                                     ).clamp(min=float(self.patience_min))

                new_exits = eligible & (pat_cnt.float() >= required_patience)
                if new_exits.any():
                    idx = new_exits.nonzero(as_tuple=True)[0].tolist()
                    for i in idx:
                        exit_lyr[i] = l
                        if self.exit_logit_smoothing and prev_int_logits is not None:
                            exit_int[i]  = (0.5 * (int_logits_l[i] + prev_int_logits[i])).detach().clone()
                            exit_slot[i] = (0.5 * (slot_logits_l[i] + prev_slot_logits[i])).detach().clone()
                        else:
                            exit_int[i]  = int_logits_l[i].detach().clone()
                            exit_slot[i] = slot_logits_l[i].detach().clone()
                    exited = exited | new_exits

            prev_ip           = ip_l.detach()
            prev_intent_pred  = intent_pred_l.detach()
            prev_slot_pred    = slot_pred_l.detach()
            prev_int_logits   = int_logits_l.detach()
            prev_slot_logits  = slot_logits_l.detach()

            if exited.all():
                break

        assert last_int_logits is not None, "No layers were iterated — backbone has no .layers?"
        for i in range(B):
            if exit_int[i] is None:
                exit_int[i]  = last_int_logits[i].detach().clone()
                exit_slot[i] = last_slot_logits[i].detach().clone()

        final_int  = torch.stack(exit_int, dim=0)
        final_slot = torch.stack(exit_slot, dim=0)
        return (final_int, final_slot), exit_lyr


# ============================================================
# 10.5  FROZEN-FEATURE CACHE  (opt-in training speedup, NOVELTY)
# ============================================================
#
# The backbone is frozen and run in eval()/no_grad() (see JointModelWith-
# EarlyExit.forward): for a fixed input, its per-layer pooled outputs
# (cls_l, word_h_l) are *exactly* deterministic across epochs -- no
# dropout, no weight updates touch it. The all-layer training objective
# (compute_loss, return_layer_probes=True) nonetheless reruns the full
# backbone forward pass every single step of every epoch, recomputing the
# identical numbers `num_train_epochs` times. Only the tiny per-layer
# heads ever change. Caching the pooled per-layer features once and
# training the heads directly off the cache turns every epoch after the
# first "backbone pass" into head-only compute, which is where almost all
# of the wall-clock/FLOPs reduction in this update comes from -- it changes
# *nothing* about what the heads see (bit-identical features, same
# gradients), so it is a pure speed optimization with no accuracy impact,
# not an approximation. This holds for --use_scl too: the SCL projection
# heads consume the exact same cached (cls_l, word_h_l) as the exit heads,
# so caching does not change the SCL loss either -- see
# EarlyExitTrainer.compute_loss_from_cache.
#
# OFF by default (`--cache_frozen_features`) because memory scales as
# N_train * num_layers * max_seq * hidden_size, which can be large for big
# backbones/datasets; a `--cache_max_gb` budget check refuses (and falls
# back to the normal path with a warning) rather than risking an OOM.

class FrozenFeatureCache:
    """
    Precomputes and stores, once, the per-layer pooled (cls_l) and
    word-aligned (word_h_l) features every exit head actually consumes --
    NOT the raw token-level hidden states (which would be `max_seq_length`
    times larger for no benefit, since only the word-pooled view is ever
    used downstream). Stored on CPU in fp16 to roughly halve memory versus
    the backbone's working dtype; moved to `device` per-batch at train time,
    same as the ordinary dataloader path already does.
    """
    def __init__(self, model: "JointModelWithEarlyExit", device: str):
        self.model  = model
        self.device = device
        self.cls_cache:  Optional[torch.Tensor] = None  # (N, L, D)  fp16 CPU
        self.word_cache: Optional[torch.Tensor] = None  # (N, L, MW, D) fp16 CPU
        self.built = False

    def estimate_bytes(self, n: int, max_words: int) -> int:
        L = self.model.num_layers
        D = next(self.model.wordrep.base_model.parameters()).shape[-1]
        # fp16 = 2 bytes; cls: N*L*D, word: N*L*MW*D
        return 2 * n * L * D * (1 + max_words)

    @torch.no_grad()
    def build(self, dataset, pad_id: int, batch_size: int,
             max_gb: float = 6.0) -> bool:
        n = len(dataset)
        max_words = dataset[0][2].shape[0] if n > 0 else 0
        est_bytes = self.estimate_bytes(n, max_words)
        est_gb = est_bytes / (1024 ** 3)
        if est_gb > max_gb:
            logger.warning(
                "FrozenFeatureCache: estimated %.2f GB exceeds --cache_max_gb=%.2f GB "
                "for %d examples. Skipping cache -- falling back to the normal "
                "(recomputed-every-epoch) training path. Raise --cache_max_gb, "
                "shrink --max_seq_length, or use a smaller train split to enable it.",
                est_gb, max_gb, n,
            )
            return False
        logger.info("FrozenFeatureCache: building for %d examples (~%.2f GB, fp16 CPU) ...",
                    n, est_gb)
        t0 = time.perf_counter()

        L = self.model.num_layers
        D = next(self.model.wordrep.base_model.parameters()).shape[-1]
        self.cls_cache  = torch.empty((n, L, D), dtype=torch.float16)
        self.word_cache = torch.empty((n, L, max_words, D), dtype=torch.float16)

        dl = DataLoader(
            dataset, sampler=SequentialSampler(dataset), num_workers=2,
            batch_size=batch_size,
            collate_fn=lambda x: collate_fn(x, pad_id),
        )
        self.model.eval()
        write_ptr = 0
        for batch in dl:
            bsz = batch[0].size(0)
            input_ids      = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            words_lengths  = batch[2].to(self.device)

            out = self.model.wordrep.base_model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True,
            )
            hs    = out.hidden_states
            align = _build_align(input_ids, words_lengths, self.device).to(dtype=hs[-1].dtype)
            for l in range(L):
                h        = hs[l + 1]
                cls_l    = self.model.wordrep.pooling(h, attention_mask)
                word_h_l = torch.bmm(align, h)
                self.cls_cache[write_ptr:write_ptr + bsz, l]  = cls_l.detach().to("cpu", torch.float16)
                self.word_cache[write_ptr:write_ptr + bsz, l] = word_h_l.detach().to("cpu", torch.float16)
            write_ptr += bsz

        self.built = True
        logger.info("FrozenFeatureCache: built in %.1fs.", time.perf_counter() - t0)
        return True


class CachedFeatureDataset(Dataset):
    """Wraps a FrozenFeatureCache + the original dataset's labels so a
    normal DataLoader/RandomSampler can shuffle cached-feature training
    batches exactly like the raw-input path does."""
    def __init__(self, base_dataset, cache: FrozenFeatureCache):
        self.base  = base_dataset
        self.cache = cache

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        _iids, _amask, _wlen, wattn, ilbl, slbl, fscr = self.base[idx]
        return (self.cache.cls_cache[idx], self.cache.word_cache[idx],
                wattn, ilbl, slbl, fscr)


def collate_fn_cached(batch):
    cls_l, word_l, wattn, ilbl, slbl, fscr = zip(*batch)
    return (
        torch.stack(cls_l),   # (B, L, D) fp16
        torch.stack(word_l),  # (B, L, MW, D) fp16
        torch.stack(wattn),
        torch.stack(ilbl),
        torch.stack(slbl),
        torch.tensor(fscr, dtype=torch.float),
    )


# ============================================================
# 11. TRAINER
# ============================================================

class EarlyExitTrainer:
    def __init__(self, args, tokenizer, train_ds, dev_ds, test_ds,
                 intent_label_set, slot_label_set,
                 intent_pos_weight: Optional[torch.Tensor] = None):
        self.args             = args
        self.device           = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer        = tokenizer
        self.trainer_state    = TrainerState()
        self.intent_label_set = intent_label_set
        self.slot_label_set   = slot_label_set
        self.train_ds         = train_ds
        self.dev_ds           = dev_ds
        self.test_ds           = test_ds
        self.model = JointModelWithEarlyExit(
            args, len(intent_label_set), len(slot_label_set)
        ).to(self.device)

        self.intent_pos_weight = (
            intent_pos_weight.to(self.device) if intent_pos_weight is not None else None
        )
        # Intent loss dispatch config (see intent_loss_func / asymmetric_loss).
        # Default "asl" is a drop-in, novelty upgrade over plain BCE+pos_weight
        # for imbalanced multi-label intent targets; pass --intent_loss_fn bce
        # to reproduce the original script's exact loss behaviour.
        self.intent_loss_fn = getattr(args, "intent_loss_fn", "asl")
        self.asl_gamma_neg  = getattr(args, "asl_gamma_neg", 4.0)
        self.asl_gamma_pos  = getattr(args, "asl_gamma_pos", 0.0)
        self.asl_clip       = getattr(args, "asl_clip", 0.05)
        if self.intent_loss_fn == "asl":
            logger.info(
                "Intent loss = Asymmetric Loss (gamma_neg=%.2f gamma_pos=%.2f clip=%.3f); "
                "pos_weight is still computed/logged above but NOT applied under ASL "
                "(ASL's own focusing + probability-shift already correct for imbalance). "
                "Pass --intent_loss_fn bce to use plain BCE+pos_weight instead.",
                self.asl_gamma_neg, self.asl_gamma_pos, self.asl_clip,
            )

        # ---- SCL config: single source of truth is self.model.use_scl -----
        # (both are constructed from the same `args`, but reading it back off
        # the already-built model rather than re-reading `args` here avoids
        # a second place this flag could silently drift out of sync with
        # what the model actually built projection heads for).
        self.use_scl           = self.model.use_scl
        self.scl_coef_intent   = getattr(args, "scl_coef_intent", 0.3)
        self.scl_coef_slot     = getattr(args, "scl_coef_slot",   0.3)
        self.scl_temp_intent   = getattr(args, "scl_temp_intent", 0.10)
        self.scl_temp_slot     = getattr(args, "scl_temp_slot",   0.10)
        self.scl_slot_max_tokens = getattr(args, "scl_slot_max_tokens", 512)
        if self.use_scl:
            logger.info(
                "SCL enabled: coef_intent=%.3f coef_slot=%.3f temp_intent=%.3f "
                "temp_slot=%.3f slot_max_tokens=%d proj_dim=%d",
                self.scl_coef_intent, self.scl_coef_slot,
                self.scl_temp_intent, self.scl_temp_slot,
                self.scl_slot_max_tokens, getattr(args, "scl_proj_dim", 128),
            )

        # Decision threshold for the multi-label intent head. Calibrated by
        # grid search on DEV inside evaluate("dev", ...) and then frozen and
        # reused for test -- never re-searched on test (would be leakage).
        self.intent_threshold = getattr(args, "intent_threshold_init", 0.5)
        self.is_multi_label = self._infer_is_multi_label(train_ds if train_ds is not None else dev_ds)
        logger.info("Detected task modality: is_multi_label=%s (used by the PABEE exit "
                    "criterion; recomputed from data, not guessed).", self.is_multi_label)

        # Fixed subsample of TRAIN, evaluated with the exact same forward
        # path + metric function as dev/test (see evaluate()'s "train_probe"
        # mode). Exists purely as a diagnostic: without it there is no
        # train-side number computed the same way as the dev numbers, so a
        # low dev accuracy next to a low training loss is ambiguous between
        # "the model is overfitting" and "the exit criterion is answering
        # from a different, worse layer than training ever evaluated."
        self.train_probe_ds = None
        if train_ds is not None:
            n = min(getattr(args, "train_probe_size", 1000), len(train_ds))
            g = torch.Generator().manual_seed(42)
            idx = torch.randperm(len(train_ds), generator=g)[:n].tolist()
            self.train_probe_ds = torch.utils.data.Subset(train_ds, idx)

        if _wandb_active:
            wb.watch(self.model, log="all",
                     log_freq=getattr(args, "wandb_watch_freq", 100), log_graph=False)

    @staticmethod
    def _infer_is_multi_label(ds, sample_size: int = 500) -> bool:
        """
        Scans up to `sample_size` examples for any utterance carrying more
        than one positive intent label. Used once at trainer construction so
        the early-exit criterion and the metric-computation path agree on
        whether to discretize intent probabilities via argmax (single-label)
        or via thresholding (multi-label / compound-intent) -- previously
        this was re-derived ad hoc, per eval call, only from whichever split
        happened to be passed in.
        """
        if ds is None:
            return True
        n = min(sample_size, len(ds))
        for i in range(n):
            if int(ds[i][4].sum().item()) > 1:
                return True
        return False

    def _dl(self, ds, shuffle):
        s = RandomSampler(ds) if shuffle else SequentialSampler(ds)
        b = self.args.train_batch_size if shuffle else self.args.eval_batch_size
        return DataLoader(
            ds, sampler=s, num_workers=4, batch_size=b,
            collate_fn=lambda x: collate_fn(x, self.tokenizer.pad_token_id),
            pin_memory=torch.cuda.is_available(), persistent_workers=True,
        )

    def _intent_loss(self, y_hat, y_true):
        """Single call-site for the intent loss so ASL/BCE selection (see
        __init__) is applied identically during training, the diagnostic
        final-layer loss, and eval -- avoids the three sites silently
        drifting out of sync."""
        return intent_loss_func(
            y_hat, y_true, pos_weight=self.intent_pos_weight,
            loss_fn=self.intent_loss_fn, asl_gamma_neg=self.asl_gamma_neg,
            asl_gamma_pos=self.asl_gamma_pos, asl_clip=self.asl_clip,
        )

    def _layer_losses(self, model, l, cls_l, word_h_l, intent_logits, slot_logits,
                      intent_labels, slot_labels, word_attention_mask):
        """
        Single source of truth for one layer's (intent + slot [+ optional SCL])
        losses. Shared by compute_loss (fresh backbone forward every step) and
        compute_loss_from_cache (cached frozen features, see FrozenFeatureCache)
        so the two training paths cannot silently drift out of sync -- both
        must produce identical loss terms given identical (cls_l, word_h_l,
        intent_logits, slot_logits) inputs.

        cls_l:    (B, D)      layer-l pooled utterance representation.
        word_h_l: (B, MW, D)  layer-l word-aligned representation.
        intent_logits, slot_logits: this layer's ExitHead outputs (already
            computed by the caller -- NOT recomputed here, to avoid a second,
            possibly-inconsistent forward through the same heads).
        """
        intent_l = self._intent_loss(intent_logits, intent_labels.float())
        s_out, s_lbl = get_useful_ones(slot_logits, slot_labels, word_attention_mask)
        slot_l = F.cross_entropy(s_out, s_lbl) if s_lbl.numel() > 0 else slot_logits.sum() * 0.0

        scl_i = torch.tensor(0.0, device=self.device)
        scl_s = torch.tensor(0.0, device=self.device)
        if self.use_scl:
            # Two independent forward passes through the SAME projection head
            # (train-mode dropout gives two different masks) = two SimCSE-
            # style positive views of the same underlying frozen feature --
            # see the module-level SCL docstring for why this replaces
            # image-style data augmentation here.
            proj_i = model.scl_intent_proj[l]
            z_int  = torch.stack([proj_i(cls_l), proj_i(cls_l)], dim=1)         # (B, 2, P)
            scl_i  = scl_intent_loss(z_int, intent_labels, temp=self.scl_temp_intent)

            proj_s = model.scl_slot_proj[l]
            z_slot = torch.stack([proj_s(word_h_l), proj_s(word_h_l)], dim=2)   # (B, MW, 2, P)
            scl_s  = scl_slot_loss(
                z_slot, slot_labels, word_attention_mask,
                temp=self.scl_temp_slot, max_tokens=self.scl_slot_max_tokens,
            )

        return intent_l, slot_l, scl_i, scl_s

    def compute_loss(self, model, inputs, slot_labels, intent_labels,
                     word_attention_mask, freq_scores: Optional[torch.Tensor] = None):
        """
        Frequency-adaptive depth weighting, applied uniformly to every
        layer's (BCE/ASL intent + CE slot [+ optional SCL]) loss -- no
        separate BiSLU/aux terms, no self-distillation.

        w_l = mean_rarity * (l+1)/L  +  (1 - mean_rarity) * (L-l)/L

        mean_rarity -> 1 (rare batch)     : deep layers weighted more
        mean_rarity -> 0 (frequent batch) : shallow layers weighted more
        mean_rarity = 0.5                 : ~uniform across depth

        The SCL terms are weighted by the SAME depth weight `w` as the
        classification terms (not a separate schedule) so a layer that is
        currently being emphasized by the frequency-adaptive weighting is
        also the layer whose representation SCL is most strongly asked to
        reshape -- keeping the two objectives aligned on which depth matters
        for the current batch, rather than fighting over different layers.
        """
        l_int, l_slot, l_cls, l_word_h = model(**inputs, return_layer_probes=True)
        L = len(l_int)

        use_freq_loss = (getattr(self.args, "use_freq_exit", False)
                         and freq_scores is not None)
        mean_rarity = float(freq_scores.mean().item()) if use_freq_loss else 0.5

        total = torch.tensor(0.0, device=self.device)
        intent_diag = torch.tensor(0.0, device=self.device)  # unweighted, for logging only
        slot_diag   = torch.tensor(0.0, device=self.device)
        scl_intent_diag = torch.tensor(0.0, device=self.device)
        scl_slot_diag   = torch.tensor(0.0, device=self.device)
        for l in range(L):
            w = (mean_rarity * (l + 1) / L
                 + (1.0 - mean_rarity) * (L - l) / L)

            intent_l, slot_l, scl_i, scl_s = self._layer_losses(
                model, l, l_cls[l], l_word_h[l], l_int[l], l_slot[l],
                intent_labels, slot_labels, word_attention_mask,
            )

            total = total + w * (
                self.args.loss_coef_intent * intent_l
                + self.args.loss_coef_slot  * slot_l
                + self.scl_coef_intent      * scl_i
                + self.scl_coef_slot        * scl_s
            )
            intent_diag = intent_diag + intent_l.detach()
            slot_diag   = slot_diag   + (slot_l.detach() if torch.is_tensor(slot_l) else slot_l)
            scl_intent_diag = scl_intent_diag + (scl_i.detach() if torch.is_tensor(scl_i) else scl_i)
            scl_slot_diag   = scl_slot_diag   + (scl_s.detach() if torch.is_tensor(scl_s) else scl_s)
        total = total / L
        intent_diag = intent_diag / L
        slot_diag   = slot_diag   / L
        scl_intent_diag = scl_intent_diag / L
        scl_slot_diag   = scl_slot_diag   / L

        # --- Diagnostic only: loss of the LAST layer's head alone. -------
        # `total` above is a mean over all L layer losses, backprop'd
        # jointly (standard PABEE-style training). Some early layers can
        # fit an easy majority-class shortcut almost immediately, which
        # pulls that mean down fast without the deep, discriminative layers
        # having learned anything yet. Comparing dev loss (computed from a
        # SINGLE exit layer's head) against `total` is therefore comparing
        # two different quantities. `final_layer_loss` uses the same head
        # (layer L-1) and the same loss terms dev/test use when the model
        # runs to full depth, so it is the correct apples-to-apples number
        # to track against dev/test loss.
        final_intent_l = self._intent_loss(l_int[-1], intent_labels.float()).detach()
        fs_out, fs_lbl = get_useful_ones(l_slot[-1], slot_labels, word_attention_mask)
        final_slot_l = (F.cross_entropy(fs_out, fs_lbl) if fs_lbl.numel() > 0
                        else l_slot[-1].sum().detach() * 0.0)
        final_layer_loss = (self.args.loss_coef_intent * final_intent_l
                            + self.args.loss_coef_slot * final_slot_l)

        return (total, l_int[-1], l_slot[-1], intent_diag, slot_diag,
                final_layer_loss, scl_intent_diag, scl_slot_diag)

    def compute_loss_from_cache(self, cls_stack, word_stack, slot_labels, intent_labels,
                                word_attention_mask, freq_scores: Optional[torch.Tensor] = None):
        """
        Identical loss computation to compute_loss(), except the per-layer
        (cls_l, word_h_l) pooled features come from a FrozenFeatureCache
        instead of a fresh backbone forward pass. Because the backbone is
        frozen/deterministic, cls_stack[:, l] / word_stack[:, l] here are
        bit-identical (up to the fp16 cache round-trip) to what
        compute_loss() would have recomputed -- only the exit-head (and,
        if --use_scl, the SCL projection-head) forward + loss + backward
        actually run, which is where the speedup comes from. Both this
        method and compute_loss() route their per-layer loss computation
        through the shared _layer_losses() helper, so SCL behaves
        identically under caching. See FrozenFeatureCache docstring.
        """
        model = self.model
        L = model.num_layers
        head_dtype = next(model.exit_intent_heads[0].parameters()).dtype
        use_freq_loss = (getattr(self.args, "use_freq_exit", False) and freq_scores is not None)
        mean_rarity = float(freq_scores.mean().item()) if use_freq_loss else 0.5

        total = torch.tensor(0.0, device=self.device)
        intent_diag = torch.tensor(0.0, device=self.device)
        slot_diag   = torch.tensor(0.0, device=self.device)
        scl_intent_diag = torch.tensor(0.0, device=self.device)
        scl_slot_diag   = torch.tensor(0.0, device=self.device)
        l_int_last, l_slot_last = None, None
        for l in range(L):
            cls_l    = cls_stack[:, l].to(head_dtype)
            word_h_l = word_stack[:, l].to(head_dtype)
            int_logits_l  = model.exit_intent_heads[l](cls_l)
            slot_logits_l = model.exit_slot_heads[l](word_h_l)
            l_int_last, l_slot_last = int_logits_l, slot_logits_l

            w = (mean_rarity * (l + 1) / L + (1.0 - mean_rarity) * (L - l) / L)
            intent_l, slot_l, scl_i, scl_s = self._layer_losses(
                model, l, cls_l, word_h_l, int_logits_l, slot_logits_l,
                intent_labels, slot_labels, word_attention_mask,
            )

            total = total + w * (
                self.args.loss_coef_intent * intent_l
                + self.args.loss_coef_slot  * slot_l
                + self.scl_coef_intent      * scl_i
                + self.scl_coef_slot        * scl_s
            )
            intent_diag = intent_diag + intent_l.detach()
            slot_diag   = slot_diag   + (slot_l.detach() if torch.is_tensor(slot_l) else slot_l)
            scl_intent_diag = scl_intent_diag + (scl_i.detach() if torch.is_tensor(scl_i) else scl_i)
            scl_slot_diag   = scl_slot_diag   + (scl_s.detach() if torch.is_tensor(scl_s) else scl_s)
        total = total / L
        intent_diag = intent_diag / L
        slot_diag   = slot_diag   / L
        scl_intent_diag = scl_intent_diag / L
        scl_slot_diag   = scl_slot_diag   / L

        final_intent_l = self._intent_loss(l_int_last, intent_labels.float()).detach()
        fs_out, fs_lbl = get_useful_ones(l_slot_last, slot_labels, word_attention_mask)
        final_slot_l = (F.cross_entropy(fs_out, fs_lbl) if fs_lbl.numel() > 0
                        else l_slot_last.sum().detach() * 0.0)
        final_layer_loss = (self.args.loss_coef_intent * final_intent_l
                            + self.args.loss_coef_slot * final_slot_l)
        return (total, l_int_last, l_slot_last, intent_diag, slot_diag,
                final_layer_loss, scl_intent_diag, scl_slot_diag)

    def _build_optimizer(self):
        no_decay   = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        trainable  = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        n_train    = sum(p.numel() for _, p in trainable)
        n_total    = sum(p.numel() for p in self.model.parameters())
        logger.info("Optimiser: %d / %d parameters (%.2f%%) — backbone frozen, "
                    "only per-layer heads (and, if enabled, SCL projection heads) are trainable.",
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
        global wb, _wandb_active

        # ---- NOVELTY: optional frozen-feature cache (see section 10.5). ----
        # Builds once, up front, then every epoch trains heads only -- no
        # backbone forward pass at all for the rest of the run. Bit-
        # identical features to the normal path (frozen + deterministic
        # backbone), so this changes speed only, never results. Off unless
        # --cache_frozen_features is passed; on failure (budget exceeded)
        # it logs a warning and falls back to the normal path automatically.
        use_cache = False
        if getattr(self.args, "cache_frozen_features", False):
            cache = FrozenFeatureCache(self.model, self.device)
            use_cache = cache.build(
                self.train_ds, self.tokenizer.pad_token_id,
                batch_size=self.args.eval_batch_size,
                max_gb=getattr(self.args, "cache_max_gb", 6.0),
            )
        if use_cache:
            cached_ds = CachedFeatureDataset(self.train_ds, cache)
            dl = DataLoader(
                cached_ds, sampler=RandomSampler(cached_ds),
                batch_size=self.args.train_batch_size,
                collate_fn=collate_fn_cached,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            dl = self._dl(self.train_ds, True)
        steps = len(dl) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        opt   = self._build_optimizer()
        sched = get_linear_schedule_with_warmup(
            opt, int(self.args.warmup_proportion * steps), steps)
        use_amp = getattr(self.args, 'use_amp', False) and torch.cuda.is_available()

        logger.info(
            "Training: steps=%d  device=%s  L=%d  min_exit=%d  patience=%d  "
            "tau_intent=%.4f  tau_slot=%.4f  AMP=%s  freq_adaptive=%s  "
            "intent_loss_fn=%s  use_scl=%s  cache_frozen_features=%s(used=%s)",
            steps, self.device, self.model.num_layers,
            self.model.min_exit_layer, self.model.patience, self.model.tau_intent,
            self.model.tau_slot, use_amp, getattr(self.args, "use_freq_exit", False),
            self.intent_loss_fn, self.use_scl,
            getattr(self.args, "cache_frozen_features", False), use_cache,
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
                "model/use_scl":        int(self.use_scl),
            }, step=0)

        es = EarlyStopping(self.args.early_stopping, verbose=True)
        self.model.zero_grad()
        gs = 0
        total_samples_seen = 0
        t_epoch_start      = time.perf_counter()

        for epoch in trange(self.args.num_train_epochs):
            self.model.train()
            ep_loss = 0.0; ep_steps = 0
            ep_intent_loss = 0.0; ep_slot_loss = 0.0
            ep_scl_intent_loss = 0.0; ep_scl_slot_loss = 0.0
            ep_final_layer_loss = 0.0
            t_step  = time.perf_counter()

            for step, batch in enumerate(dl):
                batch_size  = batch[0].size(0)
                opt.zero_grad()

                if use_cache:
                    cls_stack   = batch[0].to(self.device)
                    word_stack  = batch[1].to(self.device)
                    word_attn   = batch[2].to(self.device)
                    intent_labels = batch[3].to(self.device)
                    slot_labels   = batch[4].to(self.device)
                    freq_scores   = batch[5].to(self.device)
                    if use_amp:
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            (loss, _, _, intent_l_diag, slot_l_diag, final_layer_loss,
                             scl_intent_diag, scl_slot_diag) = self.compute_loss_from_cache(
                                cls_stack, word_stack, slot_labels, intent_labels,
                                word_attn, freq_scores=freq_scores)
                    else:
                        (loss, _, _, intent_l_diag, slot_l_diag, final_layer_loss,
                         scl_intent_diag, scl_slot_diag) = self.compute_loss_from_cache(
                            cls_stack, word_stack, slot_labels, intent_labels,
                            word_attn, freq_scores=freq_scores)
                else:
                    freq_scores = batch[6].to(self.device)
                    inputs = {
                        "input_ids":           batch[0].to(self.device),
                        "attention_mask":      batch[1].to(self.device),
                        "words_lengths":       batch[2].to(self.device),
                        "word_attention_mask": batch[3].to(self.device),
                    }
                    slot_labels   = batch[5].to(self.device)
                    intent_labels = batch[4].to(self.device)
                    word_attn     = batch[3].to(self.device)

                    if use_amp:
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            (loss, _, _, intent_l_diag, slot_l_diag, final_layer_loss,
                             scl_intent_diag, scl_slot_diag) = self.compute_loss(
                                self.model, inputs, slot_labels, intent_labels,
                                word_attn, freq_scores=freq_scores)
                    else:
                        (loss, _, _, intent_l_diag, slot_l_diag, final_layer_loss,
                         scl_intent_diag, scl_slot_diag) = self.compute_loss(
                            self.model, inputs, slot_labels, intent_labels,
                            word_attn, freq_scores=freq_scores)

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                if not torch.isfinite(loss):
                    logger.warning("Non-finite loss epoch=%d step=%d. Skipping.", epoch, step)
                    opt.zero_grad(set_to_none=True); self.model.zero_grad(set_to_none=True)
                    continue

                ep_loss += loss.item(); ep_steps += 1
                ep_intent_loss += intent_l_diag.item(); ep_slot_loss += slot_l_diag.item()
                ep_scl_intent_loss += scl_intent_diag.item(); ep_scl_slot_loss += scl_slot_diag.item()
                ep_final_layer_loss += final_layer_loss.item()
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
                            "train/intent_loss":       intent_l_diag.item(),
                            "train/slot_loss":         slot_l_diag.item(),
                            "train/scl_intent_loss":   scl_intent_diag.item(),
                            "train/scl_slot_loss":     scl_slot_diag.item(),
                            "train/intent_loss_smoothed": ep_intent_loss / max(ep_steps, 1),
                            "train/slot_loss_smoothed":   ep_slot_loss / max(ep_steps, 1),
                            "train/scl_intent_loss_smoothed": ep_scl_intent_loss / max(ep_steps, 1),
                            "train/scl_slot_loss_smoothed":   ep_scl_slot_loss / max(ep_steps, 1),
                            "train/final_layer_loss":           final_layer_loss.item(),
                            "train/final_layer_loss_smoothed":  ep_final_layer_loss / max(ep_steps, 1),
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

            epoch_loss        = ep_loss        / max(ep_steps, 1)
            epoch_intent_loss = ep_intent_loss / max(ep_steps, 1)
            epoch_slot_loss   = ep_slot_loss   / max(ep_steps, 1)
            epoch_scl_intent_loss = ep_scl_intent_loss / max(ep_steps, 1)
            epoch_scl_slot_loss   = ep_scl_slot_loss   / max(ep_steps, 1)
            epoch_final_layer_loss = ep_final_layer_loss / max(ep_steps, 1)
            epoch_time = time.perf_counter() - t_epoch_start
            t_epoch_start = time.perf_counter()
            logger.info(
                "Epoch %d done: total_loss(all-layer avg)=%.5f  final_layer_loss=%.5f  "
                "intent_loss=%.5f  slot_loss=%.5f  scl_intent_loss=%.5f  scl_slot_loss=%.5f",
                epoch, epoch_loss, epoch_final_layer_loss, epoch_intent_loss, epoch_slot_loss,
                epoch_scl_intent_loss, epoch_scl_slot_loss,
            )
            if _wandb_active:
                wb.log({"epoch/train_loss":        epoch_loss,
                        "epoch/train_final_layer_loss": epoch_final_layer_loss,
                        "epoch/train_intent_loss": epoch_intent_loss,
                        "epoch/train_slot_loss":   epoch_slot_loss,
                        "epoch/train_scl_intent_loss": epoch_scl_intent_loss,
                        "epoch/train_scl_slot_loss":   epoch_scl_slot_loss,
                        "epoch/epoch_time_sec": epoch_time,
                        "epoch/epoch": epoch}, step=gs)

            results = self.evaluate("dev", global_step=gs, epoch=epoch)

            # Dev-comparable train diagnostic: same exit-based forward path,
            # same compute_metrics call, evaluated on a fixed train subsample.
            # This is the number to actually compare against `results` above
            # -- `epoch_loss`/`epoch_final_layer_loss` are still not directly
            # comparable to dev loss because dev loss comes from whichever
            # single layer PABEE exits at, while every train loss variant
            # above is computed from a fixed layer (all-layer mean, or L-1).
            # `train_probe_results["loss"]` uses the SAME exit-selected layer
            # dev does, so it is the one apples-to-apples number: if it tracks
            # close to dev, the accuracy gap is a genuine generalization gap;
            # if it stays far below dev even on this metric, the gap is being
            # driven by exit-layer selection quality, not overfitting.
            if self.train_probe_ds is not None and (epoch % max(1, self.args.train_probe_every) == 0):
                tp_results = self.evaluate("train_probe", global_step=gs, epoch=epoch,
                                           ds_override=self.train_probe_ds)
                logger.info(
                    "  Train-probe (dev-comparable, exit-based): loss=%.5f intent_acc=%.4f "
                    "intent_micro_f1=%.4f slot_f1=%.4f mean_intent_slot=%.4f mean_f1=%.4f",
                    tp_results["loss"], tp_results["intent_acc"], tp_results["intent_micro_f1"],
                    tp_results["slot_f1"], tp_results["mean_intent_slot"], tp_results["mean_f1"],
                )

            es(results[self.args.tuning_metric], self.args)
            if es.counter == 0: self.save_model()
            if es.early_stop: logger.info("Early stopping."); break

        wb.finish()
        wb = _WandbDummy()
        _wandb_active = False

    def evaluate(self, mode="dev", global_step: int = 0, epoch: int = 0,
                ds_override=None, log_wandb: bool = True, quiet: bool = False):
        ds = ds_override if ds_override is not None else {"dev": self.dev_ds, "test": self.test_ds}.get(mode)
        if ds is None: raise ValueError(f"mode {mode!r} needs ds_override or must be 'dev'/'test'.")
        if not quiet:
            logger.info("Eval [%s] %d samples", mode, len(ds))
        dl = self._dl(ds, False)
        self.model.eval()

        ev_loss = 0.0
        int_la, int_pa, slot_la, slot_pa, mask_a = [], [], [], [], []
        all_exit_lyrs:   List[int]   = []
        all_freq_scores: List[float] = []
        layer_exit_counts = defaultdict(int)
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
                wam = batch[3].to(self.device)

                (final_i, final_s), exit_lyr_batch = self.model.forward_with_early_exit(
                    **inputs, freq_scores=freq_scores,
                    intent_threshold=self.intent_threshold,
                    is_multi_label=self.is_multi_label)

                for li in exit_lyr_batch.tolist():
                    all_exit_lyrs.append(li)
                    layer_exit_counts[li] += 1
                all_freq_scores.extend(freq_scores.cpu().tolist())

                s_out, s_lbl = get_useful_ones(final_s, sl, wam)
                slot_loss = F.cross_entropy(s_out, s_lbl) if s_lbl.numel() > 0 else final_s.sum() * 0.0
                ev_loss += (
                    self.args.loss_coef_intent * self._intent_loss(final_i, il.float())
                    + self.args.loss_coef_slot * slot_loss
                ).item()
            int_la.append(il); int_pa.append(final_i)
            slot_la.append(sl); slot_pa.append(final_s); mask_a.append(wam)

        eval_time = time.perf_counter() - t_eval_start
        ev_loss  /= len(dl)
        results   = {"loss": ev_loss}

        int_pa_cat, int_la_cat = torch.cat(int_pa, 0), torch.cat(int_la, 0)
        # Threshold calibration happens on DEV ONLY and is then frozen for
        # test/train-probe -- searching on test would be threshold leakage.
        # Uses the trainer-level cached modality flag (self.is_multi_label,
        # fixed once from a large sample at construction) rather than
        # re-deriving it from whatever subset of labels this particular
        # call happens to see, which could disagree from batch to batch on
        # a small or skewed split.
        if mode == "dev" and self.is_multi_label:
            probs_np  = torch.sigmoid(int_pa_cat.detach().float().cpu()).numpy()
            labels_np = int_la_cat.detach().cpu().numpy().astype(int)
            best_t, best_f1 = search_best_intent_threshold(probs_np, labels_np)
            if not quiet:
                logger.info(
                    "Dev intent-threshold search: best_threshold=%.2f (dev micro-F1=%.4f), "
                    "previous=%.2f", best_t, best_f1, self.intent_threshold,
                )
            self.intent_threshold = best_t

        results.update(compute_metrics(
            self.args,
            int_pa_cat, int_la_cat,
            torch.cat(slot_pa, 0), torch.cat(slot_la, 0),
            torch.cat(mask_a, 0), self.slot_label_set,
            intent_threshold=self.intent_threshold,
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

        if not quiet:
            for k in sorted(results):
                logger.info("  %-25s = %s", k, results[k])
            logger.info("  Exit: mean=%.2f std=%.2f full_pass=%.1f%% savings=%.1f%%",
                        me, se, results["pct_full_pass"]*100, results["layer_savings_pct"]*100)
            logger.info("  Exit layer distribution:")
            for li in sorted(layer_exit_counts):
                pct = 100.0 * layer_exit_counts[li] / max(len(all_exit_lyrs), 1)
                logger.info("    layer %2d : %d samples (%.1f%%)", li, layer_exit_counts[li], pct)

        if _wandb_active and log_wandb:
            prefix   = mode
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
                    m = (fs_arr >= q_edges[qi]) & (fs_arr <= q_edges[qi + 1])
                    if not m.any(): continue
                    el_q  = float(el_arr[m].mean())
                    sav_q = 1.0 - el_q / max(ml, 1)
                    strat_table.add_data(q_labels[qi], int(m.sum()),
                                         float(fs_arr[m].mean()), el_q, sav_q)
                    log_dict[f"{prefix}/freq_strat/{q_labels[qi]}/mean_exit_layer"] = el_q
                    log_dict[f"{prefix}/freq_strat/{q_labels[qi]}/layer_savings_pct"] = sav_q
                log_dict[f"{prefix}/freq_stratified_exit_table"] = strat_table
                if len(fs_arr) > 2:
                    corr = float(np.corrcoef(fs_arr, el_arr)[0, 1])
                    log_dict[f"{prefix}/rarity_exit_correlation"] = corr
                    logger.info("  Rarity-exit correlation: %.4f (expected > 0)", corr)

            log_dict.update(_gpu_mem_stats(self.device))
            log_dict["epoch/epoch"] = epoch
            wb.log(log_dict, step=global_step)

        if not quiet:
            self._write(f"eval_{mode}_results.txt", results)
        return results

    def calibrate_exit_hparams(self, patience_grid=None, min_exit_grid=None):
        """
        Grid-searches (ee_patience, min_exit_layer) on the DEV split only,
        then freezes the winning combination for the checkpoint that is
        subsequently used by evaluate("test"). This mirrors the existing
        dev-only intent-threshold search (search_best_intent_threshold)
        above, and exists for the same reason: `--ee_patience`,
        `--min_exit_layer`, `--tau_slot`, and `--intent_exit_margin` jointly
        determine which (partially-trained) layer answers for a given
        sample, and hand-picked values are not guaranteed to be anywhere
        near the accuracy/speed operating point the trained heads actually
        support. This search is cheap relative to a training run -- each
        candidate is one dev-set forward pass, no backprop, no backbone
        gradient -- so it is a low-cost way to remove one more source of
        the dev-loss/dev-accuracy mismatch you're seeing.
        """
        if patience_grid is None:
            base = self.model.patience
            patience_grid = sorted(set(max(1, v) for v in (base - 1, base, base + 1, base + 2)))
        if min_exit_grid is None:
            half = math.ceil(self.model.num_layers / 2)
            min_exit_grid = sorted(set(
                v for v in (half, half + 1, half + 2, self.model.num_layers - 1)
                if half <= v <= self.model.num_layers - 1
            ))

        base_patience, base_min_exit = self.model.patience, self.model.min_exit_layer
        best = None
        logger.info("Calibrating exit hyperparameters on DEV: patience in %s, min_exit_layer in %s",
                    patience_grid, min_exit_grid)
        for pat in patience_grid:
            for me in min_exit_grid:
                self.model.patience = pat
                self.model.min_exit_layer = me
                res = self.evaluate("dev", global_step=-1, epoch=-1, log_wandb=False, quiet=True)
                score, savings = res[self.args.tuning_metric], res["layer_savings_pct"]
                logger.info("  patience=%d min_exit=%d -> %s=%.4f  layer_savings=%.1f%%",
                            pat, me, self.args.tuning_metric, score, savings * 100)
                if best is None or score > best[0]:
                    best = (score, pat, me, savings)

        if best is None or best[0] < 0:
            logger.warning("Exit calibration found nothing better than the current settings; "
                           "keeping patience=%d min_exit_layer=%d.", base_patience, base_min_exit)
            self.model.patience, self.model.min_exit_layer = base_patience, base_min_exit
            return {"patience": base_patience, "min_exit_layer": base_min_exit}

        self.model.patience, self.model.min_exit_layer = best[1], best[2]
        logger.info("Exit calibration selected: patience=%d min_exit_layer=%d "
                    "(dev %s=%.4f, layer_savings=%.1f%%)",
                    best[1], best[2], self.args.tuning_metric, best[0], best[3] * 100)
        return {"patience": best[1], "min_exit_layer": best[2], "score": best[0]}

    def save_model(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        save_path = os.path.join(self.args.output_dir, "checkpoint.pth")
        # Only the trainable heads need to be checkpointed (backbone is
        # frozen and untouched); saving the full state_dict is still done
        # for simplicity/robustness of loading, but note most of this file
        # is the (unchanging) frozen backbone weights.
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

    intent_pos_weight = None
    if not getattr(args, "disable_intent_class_balance", False):
        intent_pos_weight = compute_intent_pos_weight(
            hf_train, int_f, is_instr, intent_label_set,
            max_weight=getattr(args, "intent_pos_weight_max", 50.0),
        )
        if _wandb_active:
            wb.log({
                "intent_pos_weight/min":  intent_pos_weight.min().item(),
                "intent_pos_weight/max":  intent_pos_weight.max().item(),
                "intent_pos_weight/mean": intent_pos_weight.mean().item(),
            }, step=0)
    else:
        logger.warning(
            "--disable_intent_class_balance set: intent BCE runs WITHOUT pos_weight. "
            "For a compound-intent, class-imbalanced target space this is very likely "
            "to reproduce the 'loss falls, subset accuracy stays at chance' pattern."
        )

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
        intent_pos_weight=intent_pos_weight,
        train_ds=make_ds(hf_train) if args.do_train else None,
        dev_ds=make_ds(hf_dev), test_ds=make_ds(hf_test),
        intent_label_set=intent_label_set, slot_label_set=slot_label_set,
    )
    if args.do_train: trainer.train()
    if args.do_eval:
        trainer.load_model()
        if getattr(args, "calibrate_exit", False):
            trainer.calibrate_exit_hparams()
        trainer.evaluate("test")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description=("Frozen-backbone joint intent detection + BIO slot filling, "
                     "with frequency-adaptive PABEE early exit and optional "
                     "per-layer supervised contrastive learning (--use_scl). "
                     "No BiSLU, no self-distillation."),
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
    p.add_argument("--learning_rate",               default=2e-4, type=float,
                   help="Only per-layer heads are trained; a higher LR than "
                        "full fine-tuning is typical for frozen-backbone probing.")
    p.add_argument("--num_train_epochs",            default=15,   type=int)
    p.add_argument("--warmup_proportion",           default=0.1,  type=float)
    p.add_argument("--gradient_accumulation_steps", default=2,    type=int)
    p.add_argument("--weight_decay",                default=0.01, type=float)
    p.add_argument("--adam_epsilon",                default=1e-8, type=float)
    p.add_argument("--max_grad_norm",               default=1.0,  type=float)
    p.add_argument("--logging_steps",               default=200,  type=int)
    p.add_argument("--early_stopping",              default=5,    type=int)
    p.add_argument("--tuning_metric",               default="mean_f1",
                   help="Default changed from 'mean_intent_slot' to 'mean_f1': "
                        "'mean_intent_slot' is (exact-match subset accuracy + slot_f1)/2, "
                        "which for a compound-intent dataset like MixATIS can sit near the "
                        "multi-label chance floor even when the intent head is learning "
                        "reasonably -- see the compute_metrics docstring. 'mean_f1' uses "
                        "intent_micro_f1 instead, which credits partially-correct predictions. "
                        "Both metrics are still logged every eval regardless of which drives "
                        "early stopping/checkpointing.")
    p.add_argument("--loss_coef_intent",     default=0.5,  type=float)
    p.add_argument("--loss_coef_slot",       default=0.5,  type=float)
    p.add_argument("--dropout_rate",   default=0.1,  type=float)
    p.add_argument("--intent_head_hidden", default=128, type=int,
                   help="Hidden width of a single non-linear layer in the per-layer intent "
                        "probe (0 = pure linear probe, matching strict linear-probing "
                        "convention). Default 128: the multi-label/compound-intent decision "
                        "boundary is harder than per-token BIO tagging, see ExitHead docstring.")
    p.add_argument("--slot_head_hidden",   default=0,   type=int,
                   help="Hidden width for the per-layer slot probe. Default 0 (pure linear): "
                        "the slot head was already performing well (F1~0.94) as a linear probe, "
                        "so it is left alone here.")
    p.add_argument("--disable_intent_class_balance", action="store_true",
                   help="Turn OFF the pos_weight class-imbalance correction on intent BCE "
                        "(for ablation only -- expect subset accuracy to collapse again).")
    p.add_argument("--intent_pos_weight_max", default=50.0, type=float,
                   help="Clip ceiling for per-class pos_weight = n_negative/n_positive.")
    p.add_argument("--intent_threshold_init", default=0.5, type=float,
                   help="Initial sigmoid threshold before the first dev-set threshold search.")
    p.add_argument("--min_exit_layer", default=None, type=int,
                   help="Hard floor: clamped up to ceil(num_layers/2) if set lower.")
    p.add_argument("--ee_patience",   default=3,    type=int)
    p.add_argument("--tau_intent",    default=0.05, type=float,
                   help="UNUSED by the current exit criterion (kept only for CLI/checkpoint "
                        "back-compat). Intent stability is now decided by discretized-label "
                        "agreement + --intent_exit_margin, not by raw probability drift -- "
                        "see forward_with_early_exit / _discretize_intent docstrings for why "
                        "the magnitude-delta version could lock onto a stable-but-wrong exit.")
    p.add_argument("--tau_slot",      default=0.1,  type=float,
                   help="Max fraction of word positions allowed to change predicted "
                        "BIO tag between consecutive layers to count as slot-stable.")
    p.add_argument("--intent_exit_margin", default=0.15, type=float,
                   help="Confidence gate for the intent exit criterion: a layer's sigmoid "
                        "output must move at least this far from --intent_threshold_init on "
                        "at least one class before consecutive-layer label agreement is "
                        "trusted as 'stable'. Prevents a flat, near-threshold, undertrained "
                        "head from satisfying patience by agreeing with itself.")
    p.add_argument("--calibrate_exit", action="store_true",
                   help="After loading the best checkpoint and before the final test "
                        "evaluation, grid-search (ee_patience, min_exit_layer) on DEV ONLY "
                        "and freeze the best combination. See calibrate_exit_hparams "
                        "docstring. No effect unless --do_eval is also set.")
    p.add_argument("--ee_patience_decay", default=0.5, type=float,
                   help="NOVELTY: shrinks the layers-of-agreement required to exit as depth "
                        "increases past the per-sample minimum (0 = original flat --ee_patience "
                        "everywhere, i.e. exact V2 behaviour). Targets the low layer_savings_pct "
                        "a flat patience produces once you're well past min_exit_layer; the "
                        "confidence-margin + joint intent/slot agreement gate is unchanged, so "
                        "this only removes conservatism, it doesn't remove the safety check.")
    p.add_argument("--ee_patience_min", default=1, type=int,
                   help="Floor on the depth-adaptive required patience above -- never requires "
                        "fewer than this many consecutive stable layers to exit.")
    p.add_argument("--disable_exit_logit_smoothing", action="store_true",
                   help="NOVELTY (on by default): the exit criterion already computes the "
                        "layer just before the one that triggers exit; averaging its logits "
                        "with the exiting layer's is a free 2-layer ensemble (zero extra FLOPs) "
                        "that reduces single-head prediction noise. Pass this flag to use only "
                        "the exiting layer's own logits (matches V2 behaviour exactly).")
    p.add_argument("--cache_frozen_features", action="store_true",
                   help="NOVELTY: precompute every layer's pooled (cls, word) features for the "
                        "frozen backbone ONCE before training, then train all epochs directly "
                        "off that cache instead of rerunning the (frozen, deterministic) "
                        "backbone forward pass every step of every epoch. Bit-identical "
                        "features -> identical gradients -> same results, purely faster. Off "
                        "by default because memory scales with dataset size x depth x "
                        "hidden_size; see --cache_max_gb. Compatible with --use_scl (SCL "
                        "projection heads consume the same cached features).")
    p.add_argument("--cache_max_gb", default=6.0, type=float,
                   help="Memory budget (GB, fp16) for --cache_frozen_features. If the train "
                        "split would exceed this, caching is skipped with a warning and "
                        "training falls back to the normal (recomputed-every-epoch) path -- "
                        "never crashes with an OOM.")
    p.add_argument("--intent_loss_fn", default="asl", choices=["asl", "bce"],
                   help="NOVELTY: intent classification loss. 'asl' = Asymmetric Loss "
                        "(Ben-Baruch et al. 2020), a multi-label-imbalance-aware replacement "
                        "for BCE+pos_weight -- see asymmetric_loss() docstring. 'bce' "
                        "reproduces the exact original BCE+pos_weight behaviour.")
    p.add_argument("--asl_gamma_neg", default=4.0, type=float,
                   help="ASL negative-class focusing exponent (higher = easy negatives matter "
                        "less). No effect if --intent_loss_fn bce.")
    p.add_argument("--asl_gamma_pos", default=0.0, type=float,
                   help="ASL positive-class focusing exponent (0 = no down-weighting of hard "
                        "positives, standard ASL default). No effect if --intent_loss_fn bce.")
    p.add_argument("--asl_clip", default=0.05, type=float,
                   help="ASL negative-probability shift; hard-discards very-easy negatives. "
                        "No effect if --intent_loss_fn bce.")
    p.add_argument("--use_scl", action="store_true",
                   help="NOVELTY: enable per-layer supervised contrastive learning (SupCon, "
                        "Khosla et al. 2020) as an auxiliary loss alongside the intent/slot "
                        "classification losses. Two dropout-based views (SimCSE, Gao et al. "
                        "2021) of each layer's frozen pooled feature are projected through a "
                        "small trainable head (ContrastiveProjectionHead) and pulled together "
                        "if their labels match (multi-label Jaccard-weighted for intent, exact "
                        "BIO-tag match for slot) and pushed apart otherwise. See the section-8.5 "
                        "module docstring for the full design rationale and known limitations "
                        "(in particular, the 'O'-tag imbalance in scl_slot_loss). Discarded at "
                        "inference -- forward_with_early_exit is unaffected.")
    p.add_argument("--scl_coef_intent", default=0.3, type=float,
                   help="Weight of the per-layer intent SCL loss in the total training loss. "
                        "No effect unless --use_scl.")
    p.add_argument("--scl_coef_slot", default=0.3, type=float,
                   help="Weight of the per-layer slot (token-level) SCL loss in the total "
                        "training loss. No effect unless --use_scl.")
    p.add_argument("--scl_temp_intent", default=0.10, type=float,
                   help="Softmax temperature for the intent SupCon loss (lower = harder "
                        "negative separation, more prone to instability). No effect unless "
                        "--use_scl.")
    p.add_argument("--scl_temp_slot", default=0.10, type=float,
                   help="Softmax temperature for the slot-level SupCon loss. No effect unless "
                        "--use_scl.")
    p.add_argument("--scl_proj_dim", default=128, type=int,
                   help="Output dimensionality of the SCL projection heads' embedding space. "
                        "No effect unless --use_scl.")
    p.add_argument("--scl_dropout_rate", default=0.1, type=float,
                   help="Dropout rate inside the SCL projection heads -- this is what supplies "
                        "the two independent 'views' per sample (SimCSE-style), so 0.0 would "
                        "make both views identical and collapse the contrastive loss to a "
                        "degenerate self-match. No effect unless --use_scl.")
    p.add_argument("--scl_slot_max_tokens", default=512, type=int,
                   help="Upper bound on valid word-token positions used per batch for the slot "
                        "SCL similarity matrix (uniformly subsampled above this, to bound the "
                        "O(N^2) cost -- see scl_slot_loss docstring). No effect unless --use_scl.")
    p.add_argument("--train_probe_size", default=1000, type=int,
                   help="Size of the fixed train subsample re-evaluated each epoch with the "
                        "exact same exit-based forward pass and metric function as dev, so "
                        "train and dev numbers are directly comparable (see train() loop).")
    p.add_argument("--train_probe_every", default=1, type=int,
                   help="Run the train-probe diagnostic every N epochs (1 = every epoch).")
    p.add_argument("--use_freq_exit", action="store_true",
                   help="Frequency-adaptive per-sample min exit + depth-weighted loss.")
    p.add_argument("--freq_smoothing",  default=0.5, type=float)
    p.add_argument("--freq_min_count",  default=1,   type=int)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--use_wandb",        action="store_true")
    p.add_argument("--wandb_project",    default="frozen-pabee-intent-slot")
    p.add_argument("--wandb_entity",     default=None)
    p.add_argument("--wandb_run_name",   default=None)
    p.add_argument("--wandb_watch_freq", default=100, type=int)

    args = p.parse_args()
    args.exit_logit_smoothing = not args.disable_exit_logit_smoothing
    if not args.do_train and not args.do_eval:
        p.error("Specify --do_train and/or --do_eval.")

    main(args)