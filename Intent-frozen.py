import os, re, sys, json, abc, warnings, logging, dataclasses, time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

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
# W&B WRAPPER  (gracefully disabled when --use_wandb not set)
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
        logger.error("wandb not installed.  Run: pip install wandb")
        return

    run_name = getattr(args, "wandb_run_name", None) or (
        f"{os.path.basename(args.model_name_or_path)}"
        f"_ee{args.ee_patience}_tau{args.tau_intent}"
        f"_frozen{args.freeze_backbone_layers if args.freeze_backbone else 'all'}"
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
        "system/gpu_mem_alloc_MB":    torch.cuda.memory_allocated(dev)     / 1024 ** 2,
        "system/gpu_mem_reserved_MB": torch.cuda.memory_reserved(dev)      / 1024 ** 2,
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
    entities, cur_type, cur_words, search_from = [], None, [], 0
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
    dataset_name, cache_dir=None, dev_split_name="validation",
    test_split_name="test", train_split_name="train", dev_fraction=0.1,
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
        utt_field = "prompt"; int_field = "completion"; slot_field = "completion"
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
        utt_field  = _pick(_U); int_field = _pick(_I); slot_field = _pick(_S)
        if utt_field is None or int_field is None:
            raise ValueError(f"Cannot detect utterance/intent fields in columns {columns}.")
        logger.info("Format: structured  utterance=%s  intent=%s  slot=%s",
                    utt_field, int_field, slot_field)

    hf_train = ds.get(train_split_name)
    hf_test  = ds.get(test_split_name)
    hf_dev   = ds.get(dev_split_name)
    if hf_train is None:
        raise ValueError(f"No '{train_split_name}' split. Available: {list(ds.keys())}")
    if hf_dev is None:
        logger.warning("No '%s' split. Carving %.0f%% of training as dev.",
                       dev_split_name, dev_fraction * 100)
        spl = hf_train.train_test_split(test_size=dev_fraction, seed=42)
        hf_train = spl["train"]; hf_dev = spl["test"]
    if hf_test is None:
        logger.warning("No '%s' split. Using dev as test.", test_split_name)
        hf_test = hf_dev
    logger.info("Split sizes  train=%d  dev=%d  test=%d",
                len(hf_train), len(hf_dev), len(hf_test))
    return hf_train, hf_dev, hf_test, utt_field, int_field, slot_field, is_instruction


def extract_label_sets(hf_train, int_field, slot_field, is_instruction):
    intent_set: Set[str] = set(); slot_type_set: Set[str] = set()
    for row in hf_train:
        if is_instruction:
            completion = row["completion"]
            for intent in parse_intents(completion).split('#'):
                intent = intent.strip()
                if intent and intent != "UNK": intent_set.add(intent)
            for tag, _ in parse_slot_pairs(completion):
                if tag.startswith('B-') and len(tag) > 2: slot_type_set.add(tag[2:])
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
                    if tag.startswith('B-') and len(tag) > 2: slot_type_set.add(tag[2:])
    intent_label_set = sorted(intent_set) + ["UNK"]
    slot_label_set   = ["_O_"] + sorted(slot_type_set) + ["UNK"]
    logger.info("Label sets: %d intents, %d slot types.",
                len(intent_label_set), len(slot_label_set))
    return intent_label_set, slot_label_set


# ============================================================
# 3.  BIO PARSING
# ============================================================

def _end_of_chunk(pt, t, pty, ty):
    if pt in ("E","S"): return True
    if pt == "B" and t in ("B","S","O"): return True
    if pt == "I" and t in ("B","S","O"): return True
    if pt not in ("O",".") and pty != ty: return True
    return False

def _start_of_chunk(pt, t, pty, ty):
    if t in ("B","S"): return True
    if pt in ("E","S","O") and t in ("E","I"): return True
    if t not in ("O",".") and pty != ty: return True
    return False

def get_bio_entities(seq, suffix=False):
    if any(isinstance(s, list) for s in seq):
        seq = [t for sub in seq for t in sub + ["O"]]
    pt, pty, begin = "O", "", 0; chunks = []
    for i, chunk in enumerate(seq + ["O"]):
        if suffix: t = chunk[-1]; ty = chunk[:-1].rsplit("-",1)[0] or "_"
        else:      t = chunk[0];  ty = chunk[1:].split("-",1)[-1] or "_"
        if _end_of_chunk(pt, t, pty, ty): chunks.append((pty, begin, i-1))
        if _start_of_chunk(pt, t, pty, ty): begin = i
        pt, pty = t, ty
    return chunks


# ============================================================
# 4.  PRECISION / RECALL / F1
# ============================================================

def _prf_divide(num, den, zero_division="warn"):
    mask = den == 0.0; den = den.copy(); den[mask] = 1; r = num / den
    if not np.any(mask): return r
    r[mask] = 0.0 if zero_division in ("warn", 0) else 1.0
    return r

def _prf(y_true, y_pred, average="micro"):
    et, ep = defaultdict(set), defaultdict(set)
    for i, yt in enumerate(y_true):
        for n, s, e in yt: et[n].add((i, s, e))
    for i, yp in enumerate(y_pred):
        for n, s, e in yp: ep[n].add((i, s, e))
    names = sorted(set(et) | set(ep))
    tp = pred = true = np.array([], dtype=np.int32)
    for n in names:
        a, b = et.get(n, set()), ep.get(n, set())
        tp   = np.append(tp, len(a & b))
        pred = np.append(pred, len(b))
        true = np.append(true, len(a))
    if average == "micro":
        tp, pred, true = np.array([tp.sum()]), np.array([pred.sum()]), np.array([true.sum()])
    prec = _prf_divide(tp, pred); rec = _prf_divide(tp, true)
    d = prec + rec; d[d == 0] = 1; f1 = 2 * prec * rec / d
    if average is not None: return np.average(prec), np.average(rec), np.average(f1)
    return prec, rec, f1

def seq_f1(yt, yp):   _, _, f = _prf(yt, yp); return f
def seq_prec(yt, yp): p, _, _ = _prf(yt, yp); return p
def seq_rec(yt, yp):  _, r, _ = _prf(yt, yp); return r

def _decode_pred(cate, scores, label_set, flat=True):
    top = [(label_set[cate[i][j].item()], i, j, scores[i][j].item())
           for i in range(len(cate)) for j in range(i, len(cate)) if cate[i][j] > 0]
    top.sort(key=lambda x: x[3], reverse=True); res = []
    for name, ns, ne, _ in top:
        for _, ts, te in res:
            if ns < ts <= ne < te or ts < ns <= te < ne: break
            if flat and (ns <= ts <= te <= ne or ts <= ns <= ne <= te): break
        else: res.append((name, ns, ne))
    return set(res)

def _decode_true(lmat, label_set):
    return [(label_set[lmat[i][j].item()], i, j)
            for i in range(len(lmat)) for j in range(i, len(lmat)) if lmat[i][j] > 0]

def get_slot_label_lists(slb, spb, wm, ls):
    yt, yp = [], []
    for i in range(len(slb)):
        tl = int(wm[i].sum().item()); p2 = spb[i][:tl, :tl]; t2 = slb[i][:tl, :tl]
        sc, c = p2.max(dim=-1)
        yp.append(list(_decode_pred(c, sc, ls))); yt.append(_decode_true(t2, ls))
    return yt, yp

def compute_metrics(args, ip, il, sp, sl, wm, ls):
    yt, yp = get_slot_label_lists(
        sl.detach().cpu(), sp.detach().float().cpu(), wm.detach().cpu(), ls)
    ip_float = ip.detach().float().cpu()
    il_cpu   = il.detach().cpu()
    single_intent = torch.all(il_cpu.sum(dim=1) == 1).item()
    if single_intent:
        pred_idx = ip_float.argmax(dim=1)
        gold_idx = il_cpu.argmax(dim=1)
        ia  = (pred_idx == gold_idx).float().mean().item()
        ipn = torch.zeros_like(il_cpu)
        ipn[torch.arange(il_cpu.size(0)), pred_idx] = 1
        ipn = ipn.numpy(); iln = il_cpu.numpy()
    else:
        probs = torch.sigmoid(ip_float)
        ipn   = (probs >= 0.3).numpy()
        iln   = il_cpu.numpy()
        ia    = accuracy_score(iln, ipn)
    sfa = float(np.mean(
        np.all(ipn == iln, axis=1) &
        np.array([set(map(tuple, p)) == set(map(tuple, t)) for p, t in zip(yp, yt)])
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
    fm  = mask.reshape(-1)
    fo  = out.reshape(-1, out.shape[-1])
    fl  = label.reshape(-1)
    idx = fm.nonzero(as_tuple=False).squeeze(-1).long()
    return fo.index_select(0, idx), fl.index_select(0, idx)

def get_soft_slot(bs, masks):
    fm = masks.reshape(-1); B, _, _, C = bs.shape
    fs = bs.reshape(-1, C).index_select(0, fm.nonzero(as_tuple=False).squeeze(-1).long())
    soft, start = [], 0
    for i in range(B):
        ln = int(masks[i].sum().item())
        soft.append(fs[start:start + ln].mean(0, keepdim=True))
        start += ln
    return torch.cat(soft, 0).to(bs.device)

def get_useful_embedding(emb, mask):
    B, n1, n2, d = emb.shape
    idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1).long()
    return emb.reshape(-1, d).index_select(0, idx)


# ============================================================
# 6.  PYTORCH DATASET
# ============================================================

class HFSLUDataset(Dataset):
    def __init__(self, args, hf_split, utterance_field, intent_field, slot_field,
                 intent_label_set, slot_label_set, tokenizer, is_instruction=True):
        self.args            = args
        self.data            = hf_split
        self.utt_field       = utterance_field
        self.int_field       = intent_field
        self.slot_field      = slot_field
        self.tokenizer       = tokenizer
        self.max_seq         = args.max_seq_length + 2
        self.intent_label_id = {w: i for i, w in enumerate(intent_label_set)}
        self.slot_label_id   = {w: i for i, w in enumerate(slot_label_set)}
        self.is_instruction  = is_instruction
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
        amask = [1] * len(iids); wattn = [1] * len(wlen)
        pad   = self.max_seq - len(wattn)
        if pad > 0: wattn += [0] * pad; wlen += [1] * pad
        return (torch.tensor(iids), torch.tensor(amask),
                torch.tensor(wlen),  torch.tensor(wattn))

    def _span_matrix(self, entities):
        starts, ends, labels = [], [], []
        for etype, es, ee in entities:
            si, ei = es + 1, ee + 1
            if si >= self.max_seq or ei >= self.max_seq: continue
            starts.append(si); ends.append(ei)
            labels.append(self.slot_label_id.get(etype, self.slot_label_id.get("UNK", 0)))
        if not starts:
            return torch.zeros(self.max_seq, self.max_seq, dtype=torch.long)
        idx = torch.tensor([starts, ends], dtype=torch.int64)
        val = torch.tensor(labels, dtype=torch.float)
        return torch.sparse.FloatTensor(
            idx, val, torch.Size([self.max_seq, self.max_seq])
        ).to_dense().long()

    def _intent_vec(self, intent_str):
        vec = [0] * len(self.intent_label_id)
        for intent in intent_str.split('#'):
            intent = intent.strip()
            idx = self.intent_label_id.get(intent, self.intent_label_id.get("UNK", 0))
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
            intent_str = ('#'.join(str(x) for x in raw_int) if isinstance(raw_int, list)
                          else str(raw_int).replace(',', ' ').replace(' ', '#'))
            if self.slot_field and row.get(self.slot_field):
                raw_s = row[self.slot_field]
                bio   = raw_s if isinstance(raw_s, list) else raw_s.split()
                ents  = get_bio_entities([str(x) for x in bio])
                slot_lbl = self._span_matrix(ents)
            else:
                slot_lbl = torch.zeros(self.max_seq, self.max_seq, dtype=torch.long)
        iids, amask, wlen, wattn = self._tokenise(words)
        int_lbl = torch.tensor(self._intent_vec(intent_str))
        return iids, amask, wlen, wattn, int_lbl, slot_lbl


def _pad_concat(tensors, pad_value=0):
    ml = max(t.size(0) for t in tensors)
    return torch.stack([F.pad(t.long(), (0, ml - t.size(0)), value=pad_value)
                        if ml > t.size(0) else t.long() for t in tensors])

def collate_fn(batch, pad_id):
    iids, amask, wlen, wattn, ilbl, slbl = zip(*batch)
    return (_pad_concat(iids, pad_id), _pad_concat(amask, 0),
            torch.stack(wlen), torch.stack(wattn),
            torch.stack(ilbl), torch.stack(slbl))


# ============================================================
# 7.  MISC UTILITIES
# ============================================================

class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience; self.verbose = verbose
        self.counter = 0; self.best_score = None; self.early_stop = False

    def __call__(self, val, args):
        s = -val if args.tuning_metric == "loss" else val
        if self.best_score is None:
            self.best_score = s; self.counter = 0
        elif s <= self.best_score:
            self.counter += 1
            if self.verbose: logger.info("EarlyStopping %d/%d", self.counter, self.patience)
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = s; self.counter = 0

@dataclass
class TrainerState:
    epoch: int = 0; global_step: int = 0; max_steps: int = 0
    num_train_epochs: int = 0; loss: float = 0.0

    def to_string(self):
        return json.dumps(dataclasses.asdict(self), sort_keys=True) + "\n"

    def save_to_json(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True))

def setup_tokenizer(model_name_or_path):
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
        logger.warning("pad_token set to eos_token ('%s').", tok.eos_token)
    return tok


# ============================================================
# 8.  LOSS FUNCTIONS
# ============================================================

class MLD(nn.Module):
    """Multi-Label Distillation self-distillation loss (Tu et al. 2023 S3.3)."""
    def forward(self, student, teacher):
        eps = 1e-9
        pS  = torch.sigmoid(student).clamp(eps, 1 - eps)
        pT  = torch.sigmoid(teacher).clamp(eps, 1 - eps)
        return (F.kl_div(pS.log(), pT, reduction="sum") +
                F.kl_div((1 - pS).log(), (1 - pT), reduction="sum")) / (student.numel() + eps)

def intent_loss_func(y_hat, y_true):
    return F.binary_cross_entropy_with_logits(y_hat.float(), y_true.float())

def probe_slot_loss_fn(logits, labels, word_mask):
    B, n, C = logits.shape
    fl, ll, ml = (logits.reshape(B * n, C),
                  labels.reshape(B * n).long(),
                  word_mask.reshape(B * n).bool())
    vl, vll = fl[ml], ll[ml]
    if vll.numel() == 0: return logits.sum() * 0.0
    return F.cross_entropy(vl, vll)

def _stable_supcon_loss(sim, pos):
    eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    sim = sim.float()
    pos = pos.bool() & (~eye)
    sim = sim.masked_fill(eye, -1e9)
    log_den  = torch.logsumexp(sim, dim=1, keepdim=True)
    log_prob = sim - log_den
    pos_count = pos.sum(dim=1)
    valid = pos_count > 0
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
    def __init__(self, d, h, dp=0.3):
        super().__init__()
        self.w1 = nn.Linear(d, h); self.w2 = nn.Linear(h, d)
        self.ln = nn.LayerNorm(d, eps=1e-6)
        self.dp1 = nn.Dropout(dp); self.dp2 = nn.Dropout(dp)

    def forward(self, x):
        r = x; x = self.ln(x)
        return r + self.dp2(self.w2(self.dp1(F.relu(self.w1(x)))))

class BiaffineLayer(nn.Module):
    def __init__(self, s1, s2, cs):
        super().__init__(); self.cs = cs
        self.bm = nn.Parameter(torch.FloatTensor(s1 + 1, cs, s2 + 1))
        nn.init.xavier_uniform_(self.bm.view(s1 + 1, -1))

    def forward(self, x1, x2):
        B, n, _ = x1.shape
        o  = torch.ones(B, n, 1, device=x1.device, dtype=x1.dtype)
        x1 = torch.cat((x1, o), -1); x2 = torch.cat((x2, o), -1)
        bl = torch.matmul(x1.reshape(-1, x1.shape[-1]), self.bm.reshape(x1.shape[-1], -1))
        bl = bl.reshape(B, n * self.cs, x2.shape[-1])
        return torch.matmul(bl, x2.transpose(1, -1)).reshape(B, n, self.cs, n).transpose(-2, -1)

class IntentClassifier(nn.Module):
    def __init__(self, d, ni, dp=0.0):
        super().__init__(); self.dp = nn.Dropout(dp); self.lin = nn.Linear(d, ni)

    def forward(self, x):
        return self.lin(self.dp(x))

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
    """Lightweight per-layer intent head used ONLY for the PABEE patience criterion."""
    def __init__(self, d, ni, dp=0.0):
        super().__init__(); self.dp = nn.Dropout(dp); self.lin = nn.Linear(d, ni)

    def forward(self, x):
        return self.lin(self.dp(x))

class EarlyExitSlotProbe(nn.Module):
    """Lightweight token-level slot probe for auxiliary training signal."""
    def __init__(self, d, ns, dp=0.0):
        super().__init__(); self.dp = nn.Dropout(dp); self.lin = nn.Linear(d, ns + 1)

    def forward(self, x):
        return self.lin(self.dp(x))

class LastTokenPooling(nn.Module):
    def forward(self, h, attn_mask):
        last = attn_mask.sum(1) - 1
        return h[torch.arange(h.shape[0], device=h.device), last]

def _build_align(input_ids, words_lengths, device):
    B, ms = input_ids.shape; mw = words_lengths.shape[1]
    align = torch.zeros(B, mw, ms)
    for i, wl in enumerate(words_lengths):
        start = 0
        for j, ln in enumerate(wl):
            ln = int(ln.item())
            if ln > 0: align[i, j, start:start + ln] = 1.0
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


# ============================================================
# 10. BACKBONE FREEZE UTILITY
# ============================================================

def _apply_backbone_freeze(base_model, num_layers: int, freeze_backbone_layers: int) -> Dict:
    """
    Freeze transformer backbone layers selectively.

    Args:
        base_model          : the HuggingFace transformer model (e.g. LlamaModel)
        num_layers          : total number of transformer layers (L)
        freeze_backbone_layers:
            -1  → freeze ALL layers (embeddings + all transformer blocks + final norm)
             0  → freeze nothing (identical to original trainable-backbone behaviour)
             k  → freeze only the first (L - k) layers; keep the top k layers trainable.
                  This is the architecturally sound compromise for a causal LM backbone:
                  the top-k layers can adapt their representations to the SLU geometry
                  while the lower layers remain as stable feature extractors.

    Returns:
        dict with 'frozen_params', 'trainable_params', 'trainable_names' for logging.
    """
    if freeze_backbone_layers == 0:
        # Nothing frozen — original behaviour
        return {
            "frozen_params":    0,
            "trainable_params": sum(p.numel() for p in base_model.parameters()),
            "trainable_names":  ["(all backbone params)"],
        }

    # --- Collect all named child modules of the transformer stack ----------
    # Works for Llama / Mistral / Phi / Qwen architectures where layers are
    # stored in model.layers.  Falls back to model.h / model.transformer.h
    # for GPT-style models.
    layer_container = None
    for attr in ("layers", "h", "blocks", "transformer"):
        candidate = getattr(base_model, attr, None)
        if candidate is not None and hasattr(candidate, "__len__"):
            layer_container = candidate
            break

    if layer_container is None:
        logger.warning(
            "Cannot locate transformer layer list in backbone.  "
            "Falling back to freezing ALL backbone parameters."
        )
        for p in base_model.parameters():
            p.requires_grad = False
        return {
            "frozen_params": sum(p.numel() for p in base_model.parameters()),
            "trainable_params": 0,
            "trainable_names": [],
        }

    # Freeze embeddings unconditionally when any freeze is applied
    for attr in ("embed_tokens", "wte", "word_embeddings", "embeddings"):
        emb = getattr(base_model, attr, None)
        if emb is not None:
            for p in emb.parameters(): p.requires_grad = False
            break

    if freeze_backbone_layers == -1:
        # Freeze everything
        for p in base_model.parameters():
            p.requires_grad = False
        frozen    = sum(p.numel() for p in base_model.parameters())
        trainable = 0
        logger.warning(
            "FULL BACKBONE FREEZE requested (freeze_backbone_layers=-1).  "
            "All %d transformer layers are frozen.  "
            "This is architecturally unsound for a raw causal LM on ATIS/MixATIS: "
            "the final-layer representations encode next-token probability, not "
            "intent/slot geometry, so a linear head cannot learn a discriminative "
            "boundary.  Expect intent_acc << random unless the backbone was already "
            "SLU-adapted.  Recommended: set freeze_backbone_layers=4 to keep the "
            "top 4 layers trainable.", num_layers
        )
        return {
            "frozen_params": frozen, "trainable_params": trainable, "trainable_names": []
        }

    # Freeze first (num_layers - freeze_backbone_layers) layers; keep top k trainable
    n_freeze = max(0, num_layers - freeze_backbone_layers)
    trainable_names = []
    for li, layer in enumerate(layer_container):
        if li < n_freeze:
            for p in layer.parameters(): p.requires_grad = False
        else:
            trainable_names.append(f"backbone.layer_{li}")

    # Keep final norm trainable when top layers are trainable
    for attr in ("norm", "ln_f", "final_layer_norm"):
        fn = getattr(base_model, attr, None)
        if fn is not None:
            for p in fn.parameters(): p.requires_grad = True
            trainable_names.append(f"backbone.{attr}")
            break

    frozen    = sum(p.numel() for p in base_model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

    logger.info(
        "Backbone freeze: %d / %d layers frozen.  Top %d layers remain trainable.  "
        "Frozen params: %d  |  Trainable backbone params: %d",
        n_freeze, num_layers, freeze_backbone_layers, frozen, trainable
    )
    return {
        "frozen_params": frozen,
        "trainable_params": trainable,
        "trainable_names": trainable_names,
    }


# ============================================================
# 11. JOINT MODEL WITH PABEE + SELECTIVE BACKBONE FREEZE
# ============================================================

class JointModelWithEarlyExit(nn.Module):
    """
    BiSLU + PABEE for decoder-only LMs with selective backbone freezing.

    Backbone freeze policy (controlled by args.freeze_backbone_layers):
    ─────────────────────────────────────────────────────────────────────
        freeze_backbone_layers = -1 : freeze ALL backbone params (full freeze)
        freeze_backbone_layers =  0 : freeze NOTHING (original trainable backbone)
        freeze_backbone_layers =  k : freeze lower (L-k) layers; keep top k trainable

    The classification heads that are ALWAYS trainable regardless of freeze policy:
        soft_intent, slot_clf, hard_intent (if use_soft_slot),
        exit_intent_heads[0..L-1], exit_slot_probes[0..L-1]

    Per-layer BiSLU auxiliary loss (layer-invariant head training):
    ─────────────────────────────────────────────────────────────────────
    The shared BiSLU head is applied at every layer's hidden state during
    training with depth weight w_l = (l+1)/L (Peters et al. 2018).
    With a frozen backbone the representations at each layer are fixed;
    the head must therefore learn a single linear boundary that works across
    all fixed layer representations simultaneously — which is only feasible
    if the backbone representations already carry SLU-discriminative structure.

    PABEE exit criterion (Xin et al. 2020, S3):
    ─────────────────────────────────────────────
    Stability-only: exit when mean|p_l - p_{l-1}| < tau_intent for
    ee_patience consecutive layers.  No confidence AND-gate.
    """

    def __init__(self, args, num_intent, num_slot):
        super().__init__()
        self.args       = args
        self.num_intent = num_intent
        self.num_slot   = num_slot

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

        # Lightweight per-layer heads — PABEE patience criterion only
        self.exit_intent_heads = nn.ModuleList([
            EarlyExitIntentHead(cfg.hidden_size, num_intent, args.dropout_rate)
            for _ in range(self.num_layers)])
        self.exit_slot_probes  = nn.ModuleList([
            EarlyExitSlotProbe(cfg.hidden_size, num_slot, args.dropout_rate)
            for _ in range(self.num_layers)])

        # Cast task heads to backbone dtype (bf16/fp16)
        _bdt = next(self.wordrep.base_model.parameters()).dtype
        for _cn, _cm in self.named_children():
            if _cn != "wordrep":
                _cm.to(dtype=_bdt)

        # ── Selective backbone freeze ─────────────────────────────────────
        freeze_layers = getattr(args, "freeze_backbone_layers", -1)
        self.freeze_info = _apply_backbone_freeze(
            self.wordrep.base_model, self.num_layers, freeze_layers
        )
        # Gradient checkpointing is meaningless when the backbone is fully frozen
        # (no backward pass through backbone → no activations to recompute).
        # Only enable it when at least some backbone layers are trainable.
        if getattr(args, 'use_gc', False):
            if freeze_layers != -1:   # at least some backbone layers are trainable
                if hasattr(self.wordrep.base_model, 'gradient_checkpointing_enable'):
                    self.wordrep.base_model.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled.")
                else:
                    logger.warning("Backbone does not support gradient_checkpointing_enable().")
            else:
                logger.info(
                    "Gradient checkpointing skipped: backbone fully frozen, "
                    "no activations need recomputation."
                )

        raw_mel             = getattr(args, "min_exit_layer", None)
        self.min_exit_layer = raw_mel if raw_mel is not None else self.num_layers // 2
        self.patience       = getattr(args, "ee_patience", 3)
        self.tau_intent     = getattr(args, "tau_intent",  0.05)
        self.tau_slot       = getattr(args, "tau_slot",    0.1)

        trainable_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_total     = sum(p.numel() for p in self.parameters())
        logger.info(
            "Model ready: L=%d  min_exit=%d  patience=%d  tau=%.4f  biaffine_dim=%d  "
            "trainable_params=%d / %d  (%.1f%%)",
            self.num_layers, self.min_exit_layer, self.patience, self.tau_intent,
            biaffine_dim, trainable_total, total_total,
            100.0 * trainable_total / max(total_total, 1),
        )

    # ── Shared BiSLU head ────────────────────────────────────────────────

    def _bislu_head(self, cls, word_h, wam):
        """Shared head applied at every layer during training (layer-invariant training)."""
        soft          = self.soft_intent(cls)
        biaffine, seg = self.slot_clf(word_h, soft, wam)
        if self.args.use_soft_slot:
            feat = self.softmax(get_soft_slot(biaffine, wam))
            hard = self.hard_intent(torch.cat([cls, feat], -1))
        else:
            hard = soft
        return cls, seg, soft, hard, biaffine

    # ── Standard forward (training) ─────────────────────────────────────

    def forward(self, input_ids, attention_mask, words_lengths,
                word_attention_mask, return_layer_probes=False):
        """
        With frozen backbone the backbone call is identical — PyTorch simply
        skips gradient accumulation for frozen parameters automatically.
        No special torch.no_grad() wrapping is needed here because
        requires_grad=False on the backbone parameters achieves the same
        effect without blocking gradient flow to the task heads.
        """
        device = input_ids.device
        out    = self.wordrep.base_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_layer_probes,
        )
        if return_layer_probes:
            hs     = out.hidden_states   # tuple: emb + L layers
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
            h     = hs[l + 1]
            cls_l = self.wordrep.pooling(h, attention_mask)
            wh_l  = torch.bmm(align, h)

            # Lightweight probe (PABEE patience signal)
            l_int.append(self.exit_intent_heads[l](cls_l))
            l_slot.append(self.exit_slot_probes[l](wh_l))

            # Shared BiSLU at layer l (layer-invariant head training)
            _, _, _, hard_l, biaffine_l = self._bislu_head(cls_l, wh_l, word_attention_mask)
            l_bislu_i.append(hard_l)
            l_bislu_s.append(biaffine_l)

        return main, l_int, l_slot, l_bislu_i, l_bislu_s

    # ── SCL augmented-view forward ────────────────────────────────────────

    def _forward_scl_embeddings(self, inputs):
        """
        Backbone is called under no_grad regardless of freeze status because
        SCL views are dropout-augmented perturbations of the frozen representation;
        no gradient should flow back to the backbone from SCL views.
        With a frozen backbone this is a no-op guard; with partial freeze it
        correctly blocks SCL gradients from reaching the backbone.
        """
        with torch.no_grad():
            out = self.wordrep.base_model(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_hidden_states=False,
            )
            h = out.last_hidden_state.detach()
        align  = _build_align(inputs['input_ids'], inputs['words_lengths'],
                               h.device).to(dtype=h.dtype)
        cls    = self.wordrep.pooling(h, inputs['attention_mask'])
        word_h = torch.bmm(align, h)
        return self._bislu_head(cls, word_h, inputs['word_attention_mask'])

    # ── PABEE early-exit inference ────────────────────────────────────────

    @torch.no_grad()
    def forward_with_early_exit(self, input_ids, attention_mask,
                                words_lengths, word_attention_mask):
        """
        Patience-Based Early Exit (Xin et al. 2020).

        Patience criterion: stability-only.
            delta_l = mean_batch_class |sigmoid(exit_head_l(h_l)) - sigmoid(exit_head_{l-1}(h_{l-1}))|
            patience_cnt increments when delta_l < tau_intent; resets otherwise.
            Exit fires when patience_cnt >= ee_patience.

        Prediction at exit: shared BiSLU head (_bislu_head) applied at exit
        layer h, which is valid because the head was trained on all layer
        representations via the per-layer auxiliary loss.
        """
        device = input_ids.device
        dummy  = next(self.wordrep.base_model.parameters())
        align  = _build_align(input_ids, words_lengths, device).to(dtype=dummy.dtype)

        out = self.wordrep.base_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hs = out.hidden_states   # [emb, h_1, ..., h_L]

        exit_layer   = self.num_layers - 1
        prev_ip      = None
        patience_cnt = 0

        for l in range(self.min_exit_layer, self.num_layers):
            h         = hs[l + 1]
            cls       = self.wordrep.pooling(h, attention_mask)
            ip_logits = self.exit_intent_heads[l](cls)
            ip_prob   = torch.sigmoid(ip_logits)

            if prev_ip is not None:
                delta = (ip_prob - prev_ip).abs().mean().item()
                if delta < self.tau_intent:
                    patience_cnt += 1
                else:
                    patience_cnt = 0
                if patience_cnt >= self.patience:
                    exit_layer = l
                    break

            prev_ip = ip_prob.detach()

        h_exit = hs[exit_layer + 1]
        bm = self.wordrep.base_model
        if hasattr(bm, "norm"):
            h_exit = bm.norm(h_exit)
        elif hasattr(bm, "ln_f"):
            h_exit = bm.ln_f(h_exit)

        cls_exit = self.wordrep.pooling(h_exit, attention_mask)
        return self._bislu_head(
            cls_exit,
            torch.bmm(align, h_exit),
            word_attention_mask,
        ), exit_layer


# ============================================================
# 12. TRAINER
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
            # Watch only trainable modules; log_graph=False avoids OOM on 1B models
            wb.watch(
                self.model,
                log="all",
                log_freq=getattr(args, "wandb_watch_freq", 100),
                log_graph=False,
            )

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

    def compute_loss(self, model, inputs, slot_labels, intent_labels, mask):
        main_out, l_int, l_slot, l_bislu_i, l_bislu_s = model(
            **inputs, return_layer_probes=True
        )
        cls_o, seg_e, soft_i, final_i, biaffine = main_out

        masks = get_mask(mask).to(self.device)
        tmp_out, tmp_lbl = get_useful_ones(biaffine, slot_labels, masks)
        ce = nn.CrossEntropyLoss(reduction="mean")

        # Main loss (final-layer BiSLU)
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

        # Auxiliary loss — two components, depth-weighted by (l+1)/L
        # Component A: lightweight exit-probe losses (PABEE patience signal)
        # Component B: shared BiSLU at layer l (layer-invariant head training)
        aux_probe = torch.tensor(0.0, device=self.device)
        aux_bislu = torch.tensor(0.0, device=self.device)

        for li in range(L):
            w = (li + 1) / L
            aux_probe = aux_probe + w * (
                self.args.loss_coef_intent * intent_loss_func(l_int[li], intent_labels.float())
                + self.args.loss_coef_slot * probe_slot_loss_fn(
                    l_slot[li], p_lbl, mask).to(self.device)
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
        """Only parameters with requires_grad=True enter the optimizer.
        Frozen backbone parameters are automatically excluded because
        _apply_backbone_freeze sets their requires_grad=False."""
        no_decay  = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        n_train   = sum(p.numel() for _, p in trainable)
        n_total   = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Optimiser: %d / %d parameters trainable (%.1f%%)  "
            "[backbone freeze_layers=%s]",
            n_train, n_total, 100.0 * n_train / max(n_total, 1),
            getattr(self.args, "freeze_backbone_layers", "N/A"),
        )
        if n_train == 0:
            raise RuntimeError(
                "No trainable parameters found.  "
                "freeze_backbone_layers=-1 froze the entire model including "
                "the task heads.  This is a misconfiguration; task heads are "
                "never frozen by _apply_backbone_freeze."
            )
        return AdamW(
            [
                {"params": [p for n, p in trainable if not any(x in n for x in no_decay)],
                 "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in trainable if     any(x in n for x in no_decay)],
                 "weight_decay": 0.0},
            ],
            lr=self.args.learning_rate, eps=self.args.adam_epsilon,
        )

    # ──────────────────────────────────────────────────────────────────────
    # TRAIN
    # ──────────────────────────────────────────────────────────────────────
    def train(self):
        dl    = self._dl(self.train_ds, True)
        steps = len(dl) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        opt   = self._build_optimizer()
        sched = get_linear_schedule_with_warmup(
            opt, int(self.args.warmup_proportion * steps), steps)
        use_amp = getattr(self.args, 'use_amp', False) and torch.cuda.is_available()

        logger.info(
            "Training: steps=%d  device=%s  L=%d  min_exit=%d  patience=%d  "
            "tau=%.4f  bislu_aux_coef=%.2f  freeze_layers=%s  AMP=%s",
            steps, self.device, self.model.num_layers,
            self.model.min_exit_layer, self.model.patience, self.model.tau_intent,
            self.args.bislu_aux_loss_coef,
            getattr(self.args, "freeze_backbone_layers", "N/A"),
            use_amp,
        )

        if _wandb_active:
            wb.log({
                "dataset/train_size":      len(self.train_ds),
                "dataset/dev_size":        len(self.dev_ds),
                "dataset/test_size":       len(self.test_ds),
                "model/num_layers":        self.model.num_layers,
                "model/num_intent":        self.model.num_intent,
                "model/num_slot":          self.model.num_slot,
                "model/min_exit_layer":    self.model.min_exit_layer,
                "model/frozen_params":     self.model.freeze_info["frozen_params"],
                "model/trainable_params":  self.model.freeze_info["trainable_params"],
                "model/freeze_backbone_layers": getattr(self.args, "freeze_backbone_layers", -1),
            }, step=0)

        es = EarlyStopping(self.args.early_stopping, verbose=True)
        self.model.zero_grad()
        gs = 0
        total_samples_seen = 0
        t_epoch_start      = time.perf_counter()

        for epoch in trange(self.args.num_train_epochs):
            self.model.train()
            ep_loss  = 0.0
            ep_steps = 0
            t_step   = time.perf_counter()

            for step, batch in enumerate(dl):
                batch_size = batch[0].size(0)
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
                            batch[5].to(self.device), batch[4].to(self.device), batch[3])
                else:
                    loss, _, _ = self.compute_loss(
                        self.model, inputs,
                        batch[5].to(self.device), batch[4].to(self.device), batch[3])

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                if not torch.isfinite(loss):
                    logger.warning("Non-finite loss epoch=%d step=%d. Skipping.", epoch, step)
                    opt.zero_grad(set_to_none=True); self.model.zero_grad(set_to_none=True)
                    continue

                ep_loss  += loss.item()
                ep_steps += 1
                loss.backward()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm)
                    if not torch.isfinite(grad_norm):
                        logger.warning("Non-finite grad norm epoch=%d step=%d.", epoch, step)
                        opt.zero_grad(set_to_none=True); self.model.zero_grad(set_to_none=True)
                        continue
                    opt.step(); sched.step(); self.model.zero_grad(set_to_none=True)
                    gs += 1
                    total_samples_seen += batch_size

                    self.trainer_state.epoch       = epoch
                    self.trainer_state.global_step = gs
                    self.trainer_state.max_steps   = steps
                    self.trainer_state.loss        = ep_loss / max(ep_steps, 1)

                    # Step-level W&B logging
                    if _wandb_active:
                        t_now   = time.perf_counter()
                        elapsed = max(t_now - t_step, 1e-6)
                        sps     = batch_size / elapsed
                        t_step  = t_now

                        step_metrics = {
                            "train/loss":              loss.item() * self.args.gradient_accumulation_steps,
                            "train/loss_smoothed":     ep_loss / max(ep_steps, 1),
                            "train/learning_rate":     sched.get_last_lr()[0],
                            "train/grad_norm":         grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm),
                            "perf/samples_per_sec":    sps,
                            "perf/total_samples_seen": total_samples_seen,
                            "train/epoch":             epoch + (step + 1) / len(dl),
                        }
                        step_metrics.update(_gpu_mem_stats(self.device))
                        wb.log(step_metrics, step=gs)

                if (step + 1) % self.args.logging_steps == 0:
                    logger.info(self.trainer_state.to_string())

            epoch_loss = ep_loss / max(ep_steps, 1)
            epoch_time = time.perf_counter() - t_epoch_start
            t_epoch_start = time.perf_counter()

            if _wandb_active:
                wb.log({
                    "epoch/train_loss":     epoch_loss,
                    "epoch/epoch_time_sec": epoch_time,
                    "epoch/epoch":          epoch,
                }, step=gs)

            results = self.evaluate("dev", global_step=gs, epoch=epoch)
            es(results[self.args.tuning_metric], self.args)
            if es.counter == 0: self.save_model()
            if es.early_stop: logger.info("Early stopping."); break

        wb.finish()

    # ──────────────────────────────────────────────────────────────────────
    # EVALUATE
    # ──────────────────────────────────────────────────────────────────────
    def evaluate(self, mode="dev", global_step: int = 0, epoch: int = 0):
        ds = {"dev": self.dev_ds, "test": self.test_ds}.get(mode)
        if ds is None: raise ValueError(f"mode must be dev or test, got {mode!r}")
        logger.info("Eval [%s] %d samples", mode, len(ds))
        dl = self._dl(ds, False)
        self.model.eval()

        ev_loss = 0.0
        int_la, int_pa, slot_la, slot_pa, mask_a, exits = [], [], [], [], [], []
        layer_exit_counts = defaultdict(int)
        ce = nn.CrossEntropyLoss(reduction="mean")
        t_eval_start = time.perf_counter()

        for batch in dl:
            with torch.no_grad():
                inputs = {
                    "input_ids":           batch[0].to(self.device),
                    "attention_mask":      batch[1].to(self.device),
                    "words_lengths":       batch[2].to(self.device),
                    "word_attention_mask": batch[3].to(self.device),
                }
                il = batch[4].to(self.device)
                sl = batch[5].to(self.device)
                out, el = self.model.forward_with_early_exit(**inputs)
                cls_o, seg_e, soft_i, final_i, biaffine = out
                exits.append(el)
                layer_exit_counts[el] += 1
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
        et = torch.tensor(exits, dtype=torch.float)
        ml = self.model.num_layers - 1
        me = et.mean().item(); se = et.std().item() if len(et) > 1 else 0.0
        results.update({
            "mean_exit_layer":   me, "std_exit_layer":    se,
            "pct_full_pass":     (et == ml).float().mean().item(),
            "layer_savings_pct": 1.0 - me / max(ml, 1),
        })

        for k in sorted(results):
            logger.info("  %-25s = %s", k, results[k])
        logger.info(
            "  Exit: mean=%.2f std=%.2f full_pass=%.1f%% savings=%.1f%%",
            me, se, results["pct_full_pass"] * 100, results["layer_savings_pct"] * 100)
        logger.info("  Exit layer distribution:")
        for li in sorted(layer_exit_counts):
            pct = 100.0 * layer_exit_counts[li] / max(len(exits), 1)
            logger.info("    layer %2d : %d batches (%.1f%%)", li, layer_exit_counts[li], pct)

        # W&B: log all eval metrics
        if _wandb_active:
            prefix   = "dev" if mode == "dev" else "test"
            log_dict = {}
            for k, v in results.items():
                log_dict[f"{prefix}/{k}"] = v
            log_dict[f"{prefix}/eval_time_sec"]   = eval_time
            log_dict[f"{prefix}/samples_per_sec"] = len(ds) / max(eval_time, 1e-6)

            if exits:
                log_dict[f"{prefix}/exit_layer_histogram"] = wb.Histogram(
                    exits, num_bins=self.model.num_layers)
                table = wb.Table(columns=["layer", "batch_count", "pct"])
                for li in range(self.model.num_layers):
                    cnt = layer_exit_counts.get(li, 0)
                    pct = 100.0 * cnt / max(len(exits), 1)
                    table.add_data(li, cnt, pct)
                log_dict[f"{prefix}/exit_layer_table"] = table

            log_dict.update(_gpu_mem_stats(self.device))
            log_dict["epoch/epoch"] = epoch
            wb.log(log_dict, step=global_step)

        self._write(f"eval_{mode}_results.txt", results)
        return results

    def save_model(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        save_path = os.path.join(self.args.output_dir, "checkpoint.pth")
        torch.save({
            "state_dict":        self.model.state_dict(),
            "intent_label_set":  self.intent_label_set,
            "slot_label_set":    self.slot_label_set,
        }, save_path)
        torch.save(self.args, os.path.join(self.args.output_dir, "training_args.bin"))
        self.trainer_state.save_to_json(os.path.join(self.args.output_dir, "trainer_state.json"))
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
# 13. MAIN + ARGPARSE
# ============================================================

def main(args):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.gpu is not None and torch.cuda.is_available():
        if args.gpu < torch.cuda.device_count():
            torch.cuda.set_device(args.gpu)
            _gp = torch.cuda.get_device_properties(args.gpu)
            print(f"GPU {args.gpu}: {_gp.name} ({_gp.total_memory / 1024 ** 3:.1f} GB)")
        else:
            raise RuntimeError(f"GPU {args.gpu} not available ({torch.cuda.device_count()} found)")
    else:
        print("CUDA not available, using CPU" if not torch.cuda.is_available()
              else f"GPU 0: {torch.cuda.get_device_properties(0).name}")

    init_wandb(args)

    hf_train, hf_dev, hf_test, utt_f, int_f, slot_f, is_instr = load_hf_dataset(
        dataset_name=args.hf_dataset, cache_dir=args.cache_dir or None,
        dev_split_name=args.dev_split, test_split_name=args.test_split,
        train_split_name=args.train_split, dev_fraction=args.dev_fraction,
    )
    intent_label_set, slot_label_set = extract_label_sets(hf_train, int_f, slot_f, is_instr)
    tokenizer = setup_tokenizer(args.model_name_or_path)

    make_ds = lambda split_data: HFSLUDataset(
        args=args, hf_split=split_data,
        utterance_field=utt_f, intent_field=int_f, slot_field=slot_f,
        intent_label_set=intent_label_set, slot_label_set=slot_label_set,
        tokenizer=tokenizer, is_instruction=is_instr,
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
        description="BiSLU + PABEE | Decoder-Only | Selective Backbone Freeze",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gpu", type=int, default=1)

    # Dataset
    p.add_argument("--hf_dataset",   required=True)
    p.add_argument("--cache_dir",    default="")
    p.add_argument("--train_split",  default="train")
    p.add_argument("--dev_split",    default="validation")
    p.add_argument("--test_split",   default="test")
    p.add_argument("--dev_fraction", default=0.1, type=float)

    # Model
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--output_dir",         required=True)

    # Training
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

    # BiSLU loss coefficients
    p.add_argument("--loss_coef_intent",     default=0.5,  type=float)
    p.add_argument("--loss_coef_slot",       default=0.5,  type=float)
    p.add_argument("--loss_coef_slot_scl",   default=0.5,  type=float)
    p.add_argument("--loss_coef_intent_scl", default=0.5,  type=float)
    p.add_argument("--sd_loss_coef",         default=0.5,  type=float)

    # BiSLU flags
    p.add_argument("--use_soft_slot",                action="store_true")
    p.add_argument("--use_scl",                      action="store_true")
    p.add_argument("--use_sd",                       action="store_true")
    p.add_argument("--use_intent_context_attention", action="store_true")
    p.add_argument("--dropout_rate",   default=0.1,  type=float)
    p.add_argument("--hidden_dim_ffw", default=300,  type=int)

    # PABEE
    p.add_argument("--min_exit_layer", default=None, type=int)
    p.add_argument("--ee_patience",    default=3,    type=int)
    p.add_argument("--tau_intent",     default=0.05, type=float)
    p.add_argument("--tau_slot",       default=0.1,  type=float)
    p.add_argument("--ee_loss_coef",   default=0.3,  type=float)
    p.add_argument("--bislu_aux_loss_coef", default=0.3, type=float)

    # Memory
    p.add_argument("--biaffine_dim", default=128, type=int)
    p.add_argument("--use_gc",  action="store_true")
    p.add_argument("--use_amp", action="store_true")

    # ── Backbone freeze control ───────────────────────────────────────────
    p.add_argument(
        "--freeze_backbone_layers", default=-1, type=int,
        help=(
            "Selective backbone freeze policy:\n"
            "  -1 : freeze ALL backbone layers (full freeze — default).\n"
            "       WARNING: architecturally unsound for raw causal LM on ATIS.\n"
            "       Use only if backbone is already SLU-adapted.\n"
            "   0 : freeze NOTHING (original fully trainable backbone).\n"
            "   k : freeze lower (L-k) layers; keep top k layers trainable.\n"
            "       Recommended: k=4 for Llama-3.2-1B (16 layers total)."
        )
    )

    # W&B
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
