import os, re, sys, gc, json, math, inspect, warnings, logging, dataclasses, time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import trange

torch.multiprocessing.set_sharing_strategy("file_system")
try:
    import resource

    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if _hard != resource.RLIM_INFINITY and _soft < _hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))
    elif _hard == resource.RLIM_INFINITY and _soft < 65536:
        resource.setrlimit(resource.RLIMIT_NOFILE, (65536, _hard))
except Exception as _e:
    logging.getLogger(__name__).warning(
        "Could not raise RLIMIT_NOFILE (%s); if 'Too many open files' recurs, raise it manually with `ulimit -n 65536` before launching.",
        _e,
    )
warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class _WandbDummy:

    def log(self, *a, **kw):
        pass

    def finish(self):
        pass

    def watch(self, *a, **kw):
        pass

    def Histogram(self, *a, **kw):
        return None

    def Table(self, *a, **kw):
        return None


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
    run_name = (
        getattr(args, "wandb_run_name", None)
        or f"{os.path.basename(args.model_name_or_path)}_frozen_ee{args.ee_patience}_tau{args.tau_intent}_minexit{args.min_exit_layer}_freqexit{int(getattr(args, 'use_freq_exit', False))}"
    )
    init_kwargs = dict(
        project=getattr(args, "wandb_project", "frozen-pabee-intent-slot"),
        entity=getattr(args, "wandb_entity", None) or None,
        name=run_name,
        config=vars(args),
        settings=_wb.Settings(_disable_stats=False, disable_code=False),
        reinit=True,
    )
    try:
        _wb.init(**init_kwargs)
        wb = _wb
        _wandb_active = True
        logger.info("W&B run initialised: %s", run_name)
        return
    except Exception as e:
        logger.warning(
            "W&B online init failed (%s: %s). This usually means the logged-in account/entity lacks write access to project '%s' (entity=%s) — check `wandb login`, the entity name, and team membership. Retrying in offline mode ...",
            type(e).__name__,
            e,
            init_kwargs["project"],
            init_kwargs["entity"],
        )
    try:
        os.environ["WANDB_MODE"] = "offline"
        _wb.init(**init_kwargs)
        wb = _wb
        _wandb_active = True
        logger.info(
            "W&B run initialised in OFFLINE mode: %s (sync later with `wandb sync`).", run_name
        )
        return
    except Exception as e:
        logger.warning(
            "W&B offline init also failed (%s: %s). Disabling W&B for this run; training will continue without it.",
            type(e).__name__,
            e,
        )
    finally:
        os.environ.pop("WANDB_MODE", None)
    wb = _WandbDummy()
    _wandb_active = False


def _gpu_mem_stats(device: str) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    dev = torch.cuda.current_device()
    return {
        "system/gpu_mem_alloc_MB": torch.cuda.memory_allocated(dev) / 1024**2,
        "system/gpu_mem_reserved_MB": torch.cuda.memory_reserved(dev) / 1024**2,
        "system/gpu_max_alloc_MB": torch.cuda.max_memory_allocated(dev) / 1024**2,
    }


def parse_utterance(prompt: str) -> List[str]:
    m = re.search("sentence:\\s*(.+?)(?:\\n|$)", prompt, re.IGNORECASE)
    text = m.group(1).strip() if m else prompt.strip()
    return text.split()


def parse_intents(completion: str) -> str:
    m = re.search("intents?:\\s*(.+?)(?:\\n|$)", completion, re.IGNORECASE)
    if not m:
        return "UNK"
    raw = m.group(1).strip()
    if "," in raw and "#" not in raw:
        return "#".join((p.strip() for p in raw.split(",") if p.strip()))
    return raw


def parse_slot_pairs(completion: str) -> List[Tuple[str, str]]:
    m = re.search("slot_labels?:\\s*(.*)", completion, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    pairs = re.findall("\\[([^\\]:]+):\\s*([^\\]]*)\\]", m.group(1))
    return [(tag.strip(), word.strip()) for tag, word in pairs if tag.strip()]


def _greedy_match(utterance_words, span_words, start_from=0):
    n = len(span_words)
    if n == 0 or start_from + n > len(utterance_words):
        return None
    for i in range(start_from, len(utterance_words) - n + 1):
        if utterance_words[i : i + n] == span_words:
            return (i, i + n - 1)
    uw = [w.lower() for w in utterance_words]
    sw = [w.lower() for w in span_words]
    for i in range(start_from, len(uw) - n + 1):
        if uw[i : i + n] == sw:
            return (i, i + n - 1)
    if start_from > 0:
        return _greedy_match(utterance_words, span_words, start_from=0)
    return None


def slot_pairs_to_entities(utterance_words, slot_pairs):
    if not slot_pairs:
        return []
    if len(slot_pairs) == len(utterance_words):
        bio_tags = [tag for tag, _ in slot_pairs]
        raw = get_bio_entities(bio_tags)
        return [(etype, s, e) for etype, s, e in raw]
    entities = []
    cur_type = None
    cur_words = []
    search_from = 0
    for tag, word in slot_pairs:
        if tag == "O" or not tag:
            if cur_type and cur_words:
                pos = _greedy_match(utterance_words, cur_words, search_from)
                if pos:
                    entities.append((cur_type, pos[0], pos[1]))
                    search_from = pos[1] + 1
            cur_type, cur_words = (None, [])
        elif tag.startswith("B-") or tag.upper() == "B":
            if cur_type and cur_words:
                pos = _greedy_match(utterance_words, cur_words, search_from)
                if pos:
                    entities.append((cur_type, pos[0], pos[1]))
                    search_from = pos[1] + 1
            cur_type = tag[2:] if tag.startswith("B-") else "_"
            cur_words = [word]
        elif tag.startswith("I-") or tag.upper() == "I":
            if cur_type is not None:
                cur_words.append(word)
            else:
                cur_type = tag[2:] if tag.startswith("I-") else "_"
                cur_words = [word]
    if cur_type and cur_words:
        pos = _greedy_match(utterance_words, cur_words, search_from)
        if pos:
            entities.append((cur_type, pos[0], pos[1]))
    return entities


def load_hf_dataset(
    dataset_name,
    cache_dir=None,
    dev_split_name="validation",
    test_split_name="test",
    train_split_name="train",
    dev_fraction=0.1,
    test_fraction=0.1,
):
    from datasets import load_dataset

    logger.info("Loading '%s' from HuggingFace Hub ...", dataset_name)
    try:
        ds = load_dataset(dataset_name, cache_dir=cache_dir)
    except FileNotFoundError as e:
        logger.warning(
            "load_dataset() failed with FileNotFoundError (%s). Falling back to direct parquet download via huggingface_hub ...",
            e,
        )
        from huggingface_hub import HfApi, hf_hub_download
        from datasets import Dataset, DatasetDict

        api = HfApi()
        repo_files = api.list_repo_files(dataset_name, repo_type="dataset")
        parquet_files = [f for f in repo_files if f.endswith(".parquet")]
        if not parquet_files:
            raise RuntimeError(
                f"Fallback failed: no parquet files found in repo '{dataset_name}'."
            ) from e
        logger.info("Fallback: found %d parquet file(s): %s", len(parquet_files), parquet_files)
        splits = {}
        for f in parquet_files:
            local_path = hf_hub_download(
                repo_id=dataset_name, filename=f, repo_type="dataset", cache_dir=cache_dir
            )
            base = f.split("/")[-1]
            if base.startswith(train_split_name):
                split_key = train_split_name
            elif base.startswith(test_split_name):
                split_key = test_split_name
            elif base.startswith(dev_split_name):
                split_key = dev_split_name
            elif "valid" in base.lower():
                split_key = dev_split_name
            else:
                split_key = train_split_name
            logger.info("Fallback: loading %s -> split '%s'", local_path, split_key)
            splits[split_key] = Dataset.from_parquet(local_path)
        ds = DatasetDict(splits)
    ref_split = train_split_name if train_split_name in ds else list(ds.keys())[0]
    columns = list(ds[ref_split].features.keys())
    sample = dict(ds[ref_split][0])
    logger.info("=" * 60)
    logger.info("Dataset columns  : %s", columns)
    logger.info("Available splits : %s", list(ds.keys()))
    for k, v in sample.items():
        logger.info("  %-20s = %r", k, str(v)[:120])
    logger.info("=" * 60)
    if "prompt" in columns and "completion" in columns:
        is_instruction = True
        utt_field = "prompt"
        int_field = "completion"
        slot_field = "completion"
        logger.info("Format: instruction-tuning (prompt/completion).")
    else:
        is_instruction = False
        _U = ["text", "utterance", "sentence", "input", "seq_in", "tokens", "words"]
        _I = ["intents", "intent_label", "intent", "label", "labels"]
        _S = ["slots", "slot_label", "seq_out", "slot_tags", "ner_tags", "bio_tags"]

        def _pick(aliases):
            for a in aliases:
                if a in columns:
                    return a
            return None

        utt_field = _pick(_U)
        int_field = _pick(_I)
        slot_field = _pick(_S)
        if utt_field is None or int_field is None:
            raise ValueError(f"Cannot detect utterance/intent fields in columns {columns}.")
        logger.info(
            "Format: structured  utterance=%s  intent=%s  slot=%s", utt_field, int_field, slot_field
        )
    hf_train = ds.get(train_split_name)
    hf_test = ds.get(test_split_name)
    hf_dev = ds.get(dev_split_name)
    if hf_train is None:
        raise ValueError(f"No '{train_split_name}' split. Available: {list(ds.keys())}")
    if hf_test is None:
        logger.info("No test split. Carving %.1f%% of training as TEST.", test_fraction * 100)
        tmp = hf_train.train_test_split(test_size=test_fraction, seed=42)
        hf_train = tmp["train"]
        hf_test = tmp["test"]
        logger.info("After test carve: train=%d, test=%d", len(hf_train), len(hf_test))
    if hf_dev is None:
        logger.info(
            "No dev split. Carving %.1f%% of remaining training as DEV.", dev_fraction * 100
        )
        tmp = hf_train.train_test_split(test_size=dev_fraction, seed=42)
        hf_train = tmp["train"]
        hf_dev = tmp["test"]
        logger.info("After dev carve: train=%d, dev=%d", len(hf_train), len(hf_dev))
    if "id" in hf_train.column_names:
        tr_ids = set(hf_train["id"])
        dv_ids = set(hf_dev["id"])
        te_ids = set(hf_test["id"])
        overlaps = (tr_ids & dv_ids, tr_ids & te_ids, dv_ids & te_ids)
        labels = ("Train-Dev", "Train-Test", "Dev-Test")
        if any(overlaps):
            for lbl, ov in zip(labels, overlaps):
                if ov:
                    logger.error("DATA LEAKAGE: %s overlap: %d samples", lbl, len(ov))
            raise RuntimeError("Dataset splits overlap! Check carving logic.")
        else:
            logger.info("No leakage detected between train/dev/test splits.")
    logger.info("=" * 60)
    logger.info(
        "FINAL SPLIT SIZES:  train=%d  dev=%d  test=%d", len(hf_train), len(hf_dev), len(hf_test)
    )
    logger.info("=" * 60)
    return (hf_train, hf_dev, hf_test, utt_field, int_field, slot_field, is_instruction)


def extract_label_sets(hf_train, int_field, slot_field, is_instruction):
    intent_set: Set[str] = set()
    slot_type_set: Set[str] = set()
    for row in hf_train:
        if is_instruction:
            completion = row["completion"]
            for intent in parse_intents(completion).split("#"):
                intent = intent.strip()
                if intent and intent != "UNK":
                    intent_set.add(intent)
            for tag, _ in parse_slot_pairs(completion):
                if tag.startswith("B-") and len(tag) > 2:
                    slot_type_set.add(tag[2:])
        else:
            raw_int = row[int_field]
            if isinstance(raw_int, list):
                for x in raw_int:
                    intent_set.add(str(x).strip())
            else:
                for x in str(raw_int).replace(",", "#").split("#"):
                    if x.strip():
                        intent_set.add(x.strip())
            if slot_field:
                raw_s = row[slot_field]
                tags = raw_s if isinstance(raw_s, list) else raw_s.split()
                for tag in tags:
                    tag = str(tag)
                    if tag.startswith("B-") and len(tag) > 2:
                        slot_type_set.add(tag[2:])
    intent_label_set = sorted(intent_set) + ["UNK"]
    slot_label_set = ["O"] + [f"{p}-{t}" for t in sorted(slot_type_set) for p in ("B", "I")]
    logger.info(
        "Label sets: %d intents, %d BIO slot tags (%d entity types).",
        len(intent_label_set),
        len(slot_label_set),
        len(slot_type_set),
    )
    return (intent_label_set, slot_label_set)


class WordFrequencyIndex:
    _STOPWORDS: Set[str] = frozenset(
        {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "i",
            "me",
            "my",
            "we",
            "our",
            "you",
            "your",
            "he",
            "she",
            "it",
            "they",
            "do",
            "does",
            "did",
            "to",
            "of",
            "in",
            "on",
            "at",
            "for",
            "with",
            "and",
            "or",
            "but",
            "not",
            "what",
            "which",
            "who",
            "how",
            "when",
            "where",
            "why",
            "can",
            "could",
            "will",
            "would",
            "should",
            "shall",
            "may",
            "might",
            "please",
            "want",
            "need",
            "find",
            "show",
            "tell",
            "get",
            "give",
            "make",
            "like",
            "from",
            "about",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
        }
    )

    def __init__(self, smoothing: float = 0.5, min_freq: int = 1):
        self.smoothing = smoothing
        self.min_freq = min_freq
        self._counts: Dict[str, int] = {}
        self._log_freq: Dict[str, float] = {}
        self._max_log_freq: float = 1.0
        self._built = False

    def build(self, hf_split, utterance_field: str, is_instruction: bool) -> None:
        counts: Dict[str, int] = defaultdict(int)
        for row in hf_split:
            if is_instruction:
                words = parse_utterance(row["prompt"])
            else:
                raw = row[utterance_field]
                words = raw if isinstance(raw, list) else raw.split()
            for w in words:
                counts[w.lower()] += 1
        self._counts = {w: c for w, c in counts.items() if c >= self.min_freq}
        self._log_freq = {w: math.log(c + self.smoothing) for w, c in self._counts.items()}
        self._max_log_freq = max(self._log_freq.values(), default=1.0)
        self._built = True
        logger.info(
            "WordFrequencyIndex built: %d unique words | max_freq=%d | max_log_freq=%.3f",
            len(self._counts),
            max(self._counts.values(), default=0),
            self._max_log_freq,
        )

    def word_log_freq_norm(self, word: str) -> float:
        raw = self._log_freq.get(word.lower(), math.log(self.smoothing))
        return max(raw, 0.0) / max(self._max_log_freq, 1e-09)

    def utterance_rarity_score(self, words: List[str]) -> float:
        if not self._built:
            return 0.5
        content = [w for w in words if w.lower() not in self._STOPWORDS]
        if not content:
            content = words
        min_norm_lf = min((self.word_log_freq_norm(w) for w in content))
        return float(np.clip(1.0 - min_norm_lf, 0.0, 1.0))

    def summary_stats(self) -> Dict[str, float]:
        if not self._counts:
            return {}
        counts_arr = np.array(list(self._counts.values()), dtype=np.float64)
        return {
            "freq_index/vocab_size": len(self._counts),
            "freq_index/mean_freq": float(counts_arr.mean()),
            "freq_index/median_freq": float(np.median(counts_arr)),
            "freq_index/max_freq": float(counts_arr.max()),
            "freq_index/singleton_pct": float((counts_arr == 1).mean() * 100),
        }


def _end_of_chunk(pt, t, pty, ty):
    if pt in ("E", "S"):
        return True
    if pt == "B" and t in ("B", "S", "O"):
        return True
    if pt == "I" and t in ("B", "S", "O"):
        return True
    if pt not in ("O", ".") and pty != ty:
        return True
    return False


def _start_of_chunk(pt, t, pty, ty):
    if t in ("B", "S"):
        return True
    if pt in ("E", "S", "O") and t in ("E", "I"):
        return True
    if t not in ("O", ".") and pty != ty:
        return True
    return False


def get_bio_entities(seq, suffix=False):
    if any((isinstance(s, list) for s in seq)):
        seq = [t for sub in seq for t in sub + ["O"]]
    pt, pty, begin = ("O", "", 0)
    chunks = []
    for i, chunk in enumerate(seq + ["O"]):
        if suffix:
            t = chunk[-1]
            ty = chunk[:-1].rsplit("-", 1)[0] or "_"
        else:
            t = chunk[0]
            ty = chunk[1:].split("-", 1)[-1] or "_"
        if _end_of_chunk(pt, t, pty, ty):
            chunks.append((pty, begin, i - 1))
        if _start_of_chunk(pt, t, pty, ty):
            begin = i
        pt, pty = (t, ty)
    return chunks


def _prf_divide(num, den, zero_division="warn"):
    mask = den == 0.0
    den = den.copy()
    den[mask] = 1
    r = num / den
    if not np.any(mask):
        return r
    r[mask] = 0.0 if zero_division in ("warn", 0) else 1.0
    return r


def _prf(y_true, y_pred, average="micro"):
    et, ep = (defaultdict(set), defaultdict(set))
    for i, yt in enumerate(y_true):
        for n, s, e in yt:
            et[n].add((i, s, e))
    for i, yp in enumerate(y_pred):
        for n, s, e in yp:
            ep[n].add((i, s, e))
    names = sorted(set(et) | set(ep))
    tp = pred = true = np.array([], dtype=np.int32)
    for n in names:
        a, b = (et.get(n, set()), ep.get(n, set()))
        tp = np.append(tp, len(a & b))
        pred = np.append(pred, len(b))
        true = np.append(true, len(a))
    if average == "micro":
        tp, pred, true = (np.array([tp.sum()]), np.array([pred.sum()]), np.array([true.sum()]))
    prec = _prf_divide(tp, pred)
    rec = _prf_divide(tp, true)
    d = prec + rec
    d[d == 0] = 1
    f1 = 2 * prec * rec / d
    if average is not None:
        return (np.average(prec), np.average(rec), np.average(f1))
    return (prec, rec, f1)


def seq_f1(yt, yp):
    _, _, f = _prf(yt, yp)
    return f


def seq_prec(yt, yp):
    p, _, _ = _prf(yt, yp)
    return p


def seq_rec(yt, yp):
    _, r, _ = _prf(yt, yp)
    return r


def get_slot_label_lists(slot_label_ids, slot_logits, word_attention_mask, label_set):
    pred_ids = slot_logits.argmax(dim=-1)
    yt, yp = ([], [])
    for i in range(len(slot_label_ids)):
        tl = int(word_attention_mask[i].sum().item())
        gold_tags = [label_set[int(x)] for x in slot_label_ids[i][:tl].tolist()]
        pred_tags = [label_set[int(x)] for x in pred_ids[i][:tl].tolist()]
        yt.append(get_bio_entities(gold_tags))
        yp.append(get_bio_entities(pred_tags))
    return (yt, yp)


def compute_balanced_score(results: Dict, args) -> float:
    w_intent = getattr(args, "balanced_weight_intent_acc", 0.3)
    w_f1 = getattr(args, "balanced_weight_mean_f1", 0.25)
    w_sem = getattr(args, "balanced_weight_semantic_acc", 0.2)
    w_eff = getattr(args, "balanced_weight_efficiency", 0.25)
    total_w = max(w_intent + w_f1 + w_sem + w_eff, 1e-08)
    intent_acc = results.get("intent_acc", 0.0)
    mean_f1 = results.get("mean_f1", 0.0)
    semantic_acc = results.get("semantic_acc", 0.0)
    layer_savings_pct = results.get("layer_savings_pct", 0.0)
    return (
        w_intent * intent_acc + w_f1 * mean_f1 + w_sem * semantic_acc + w_eff * layer_savings_pct
    ) / total_w


def compute_metrics(args, ip, il, sp, sl, wm, ls, intent_threshold: float = 0.5):
    yt, yp = get_slot_label_lists(
        sl.detach().cpu(), sp.detach().float().cpu(), wm.detach().cpu(), ls
    )
    ip_float = ip.detach().float().cpu()
    il_cpu = il.detach().cpu()
    single_intent = torch.all(il_cpu.sum(dim=1) == 1).item()
    if single_intent:
        pred_idx = ip_float.argmax(dim=1)
        gold_idx = il_cpu.argmax(dim=1)
        ipn = torch.zeros_like(il_cpu)
        ipn[torch.arange(il_cpu.size(0)), pred_idx] = 1
        ipn = ipn.numpy().astype(int)
        iln = il_cpu.numpy().astype(int)
    else:
        probs = torch.sigmoid(ip_float)
        ipn = (probs >= intent_threshold).numpy().astype(int)
        iln = il_cpu.numpy().astype(int)
    ia = accuracy_score(iln, ipn)
    p_mi, r_mi, f_mi, _ = precision_recall_fscore_support(
        iln, ipn, average="micro", zero_division=0
    )
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(
        iln, ipn, average="macro", zero_division=0
    )
    sfa = float(
        np.mean(
            np.all(ipn == iln, axis=1)
            & np.array([set(map(tuple, p)) == set(map(tuple, t)) for p, t in zip(yp, yt)])
        )
    )
    f = seq_f1(yt, yp)
    return {
        "intent_acc": ia,
        "intent_micro_f1": f_mi,
        "intent_micro_precision": p_mi,
        "intent_micro_recall": r_mi,
        "intent_macro_f1": f_ma,
        "intent_threshold_used": intent_threshold,
        "slot_precision": seq_prec(yt, yp),
        "slot_recall": seq_rec(yt, yp),
        "slot_f1": f,
        "mean_intent_slot": (ia + f) / 2.0,
        "mean_f1": (f_mi + f) / 2.0,
        "semantic_acc": sfa,
    }


def search_best_intent_threshold(
    probs: np.ndarray, labels: np.ndarray, grid: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = (0.5, -1.0)
    for t in grid:
        preds = (probs >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, average="micro", zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_t = (f1, float(t))
    return (best_t, best_f1)


def get_useful_ones(out, label, mask):
    fm = mask.reshape(-1).bool()
    fo = out.reshape(-1, out.shape[-1])
    fl = label.reshape(-1)
    idx = fm.nonzero(as_tuple=False).squeeze(-1).long()
    return (fo.index_select(0, idx), fl.index_select(0, idx))


class HFSLUDataset(Dataset):

    def __init__(
        self,
        args,
        hf_split,
        utterance_field,
        intent_field,
        slot_field,
        intent_label_set,
        slot_label_set,
        tokenizer,
        is_instruction=True,
        freq_index: Optional[WordFrequencyIndex] = None,
    ):
        self.args = args
        self.data = hf_split
        self.utt_field = utterance_field
        self.int_field = intent_field
        self.slot_field = slot_field
        self.tokenizer = tokenizer
        self.max_seq = args.max_seq_length + 2
        self.intent_label_id = {w: i for i, w in enumerate(intent_label_set)}
        self.slot_label_id = {w: i for i, w in enumerate(slot_label_set)}
        self.o_id = self.slot_label_id.get("O", 0)
        self.is_instruction = is_instruction
        self.freq_index = freq_index
        self._has_bos = (
            tokenizer.bos_token is not None and tokenizer.bos_token != tokenizer.eos_token
        )
        self._has_eos = tokenizer.eos_token is not None

    def _tokenise(self, words):
        tokens, wlen = ([], [])
        if self._has_bos:
            tokens.append(self.tokenizer.bos_token)
            wlen.append(1)
        for w in words:
            toks = self.tokenizer.tokenize(w) or [self.tokenizer.unk_token]
            tokens.extend(toks)
            wlen.append(len(toks))
        if self._has_eos:
            tokens.append(self.tokenizer.eos_token)
            wlen.append(1)
        iids = self.tokenizer.convert_tokens_to_ids(tokens)
        amask = [1] * len(iids)
        wattn = [1] * len(wlen)
        pad = self.max_seq - len(wattn)
        if pad > 0:
            wattn += [0] * pad
            wlen += [1] * pad
        return (torch.tensor(iids), torch.tensor(amask), torch.tensor(wlen), torch.tensor(wattn))

    def _bio_label_seq(self, entities):
        labels = torch.full((self.max_seq,), self.o_id, dtype=torch.long)
        for etype, es, ee in entities:
            si, ei = (es + 1, ee + 1)
            if si >= self.max_seq:
                continue
            ei = min(ei, self.max_seq - 1)
            b_id = self.slot_label_id.get(f"B-{etype}")
            i_id = self.slot_label_id.get(f"I-{etype}")
            if b_id is None:
                continue
            labels[si] = b_id
            if i_id is not None and ei > si:
                labels[si + 1 : ei + 1] = i_id
        return labels

    def _intent_vec(self, intent_str):
        vec = [0] * len(self.intent_label_id)
        for intent in intent_str.split("#"):
            intent = intent.strip()
            idx = self.intent_label_id.get(intent, self.intent_label_id.get("UNK", 0))
            vec[idx] = 1
        return vec

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        if self.is_instruction:
            words = parse_utterance(row["prompt"])
            intent_str = parse_intents(row["completion"])
            slot_pairs = parse_slot_pairs(row["completion"])
            if len(words) > self.args.max_seq_length:
                words = words[: self.args.max_seq_length]
            entities = slot_pairs_to_entities(words, slot_pairs)
            slot_lbl = self._bio_label_seq(entities)
        else:
            raw_utt = row[self.utt_field]
            words = raw_utt if isinstance(raw_utt, list) else raw_utt.split()
            if len(words) > self.args.max_seq_length:
                words = words[: self.args.max_seq_length]
            raw_int = row[self.int_field]
            intent_str = (
                "#".join((str(x) for x in raw_int))
                if isinstance(raw_int, list)
                else str(raw_int).replace(",", " ").replace(" ", "#")
            )
            if self.slot_field and row.get(self.slot_field):
                raw_s = row[self.slot_field]
                bio = raw_s if isinstance(raw_s, list) else raw_s.split()
                ents = get_bio_entities([str(x) for x in bio])
                slot_lbl = self._bio_label_seq(ents)
            else:
                slot_lbl = torch.full((self.max_seq,), self.o_id, dtype=torch.long)
        iids, amask, wlen, wattn = self._tokenise(words)
        int_lbl = torch.tensor(self._intent_vec(intent_str))
        freq_score = (
            self.freq_index.utterance_rarity_score(words) if self.freq_index is not None else 0.5
        )
        return (iids, amask, wlen, wattn, int_lbl, slot_lbl, freq_score)


def _pad_concat(tensors, pad_value=0):
    ml = max((t.size(0) for t in tensors))
    return torch.stack(
        [
            F.pad(t.long(), (0, ml - t.size(0)), value=pad_value) if ml > t.size(0) else t.long()
            for t in tensors
        ]
    )


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


class EarlyStopping:

    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val, args):
        s = -val if args.tuning_metric == "loss" else val
        if self.best_score is None:
            self.best_score = s
            self.counter = 0
        elif s <= self.best_score:
            self.counter += 1
            if self.verbose:
                logger.info("EarlyStopping %d/%d", self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = s
            self.counter = 0


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    max_steps: int = 0
    num_train_epochs: int = 0
    loss: float = 0.0

    def to_string(self):
        return json.dumps(dataclasses.asdict(self), sort_keys=True) + "\n"

    def save_to_json(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True))


def setup_tokenizer(model_name_or_path):
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        logger.warning("pad_token set to eos_token ('%s').", tok.eos_token)
    return tok


def asymmetric_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    gamma_neg: float = 4.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
    eps: float = 1e-08,
) -> torch.Tensor:
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


def intent_loss_func(
    y_hat,
    y_true,
    pos_weight: Optional[torch.Tensor] = None,
    loss_fn: str = "bce",
    asl_gamma_neg: float = 4.0,
    asl_gamma_pos: float = 0.0,
    asl_clip: float = 0.05,
):
    if loss_fn == "asl":
        return asymmetric_loss(
            y_hat, y_true, gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip
        )
    return F.binary_cross_entropy_with_logits(y_hat.float(), y_true.float(), pos_weight=pos_weight)


def compute_intent_pos_weight(
    hf_train, int_field, is_instruction, intent_label_set, max_weight: float = 50.0
) -> torch.Tensor:
    n = len(intent_label_set)
    counts = np.zeros(n, dtype=np.float64)
    total = 0
    idx_of = {w: i for i, w in enumerate(intent_label_set)}
    for row in hf_train:
        total += 1
        if is_instruction:
            intents = parse_intents(row["completion"]).split("#")
        else:
            raw_int = row[int_field]
            intents = (
                [str(x) for x in raw_int]
                if isinstance(raw_int, list)
                else str(raw_int).replace(",", "#").split("#")
            )
        for it in intents:
            it = it.strip()
            if not it:
                continue
            j = idx_of.get(it, idx_of.get("UNK"))
            if j is not None:
                counts[j] += 1
    pos = np.clip(counts, 1.0, None)
    neg = np.clip(total - counts, 0.0, None)
    pw = np.clip(neg / pos, 0.1, max_weight)
    logger.info(
        "Intent pos_weight (class-imbalance correction): min=%.2f max=%.2f mean=%.2f median=%.2f (clipped to [0.1, %.1f]); %d/%d classes hit the upper clip.",
        pw.min(),
        pw.max(),
        pw.mean(),
        float(np.median(pw)),
        max_weight,
        int((pw >= max_weight).sum()),
        n,
    )
    return torch.tensor(pw, dtype=torch.float32)


class ExitHead(nn.Module):

    def __init__(
        self, hidden_size: int, num_labels: int, dropout_rate: float = 0.1, hidden_dim: int = 0
    ):
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
            self.net = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(hidden_size, num_labels))

    def forward(self, x):
        return self.net(self.norm(x))


class LastTokenPooling(nn.Module):

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B = attn_mask.shape[0]
        last = (attn_mask.sum(dim=1) - 1).clamp(0, h.shape[1] - 1)
        return h[torch.arange(B, device=h.device), last]


def _build_align(input_ids, words_lengths, device):
    B, ms = input_ids.shape
    mw = words_lengths.shape[1]
    align = torch.zeros(B, mw, ms)
    for i, wl in enumerate(words_lengths):
        start = 0
        for j, ln in enumerate(wl):
            ln = int(ln.item())
            if ln > 0:
                align[i, j, start : start + ln] = 1.0
            start += ln
    return align.to(device)


class DecoderWordRep(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(args.model_name_or_path)
        self.pooling = LastTokenPooling()

    def forward(self, input_ids, attention_mask, words_lengths):
        with torch.no_grad():
            out = self.base_model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
        h_sub = out.last_hidden_state
        align = _build_align(input_ids, words_lengths, h_sub.device).to(dtype=h_sub.dtype)
        return (self.pooling(h_sub, attention_mask), torch.bmm(align, h_sub), out.hidden_states)


def _build_causal_mask(attention_mask: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    B, T = attention_mask.shape
    dtype = hidden_states.dtype
    device = hidden_states.device
    min_v = torch.finfo(dtype).min
    causal = torch.triu(torch.full((T, T), min_v, device=device, dtype=dtype), diagonal=1)
    pad = ((1.0 - attention_mask.float()) * min_v).to(dtype)
    return causal[None, None, :, :] + pad[:, None, None, :]


def _layer_kwargs_for(
    layer: nn.Module,
    causal_mask: Optional[torch.Tensor],
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    position_embeddings: Optional[Tuple],
) -> Dict:
    sig = inspect.signature(layer.forward)
    params = sig.parameters
    has_var_kw = any((p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()))
    candidates: Dict[str, object] = {
        "attention_mask": causal_mask,
        "position_ids": position_ids,
        "past_key_value": None,
        "past_key_values": None,
        "output_attentions": False,
        "use_cache": False,
        "cache_position": cache_position,
    }
    if position_embeddings is not None:
        candidates["position_embeddings"] = position_embeddings
    if has_var_kw:
        return candidates
    return {k: v for k, v in candidates.items() if k in params}


class JointModelWithEarlyExit(nn.Module):

    def __init__(self, args, num_intent, num_slot):
        super().__init__()
        self.args = args
        self.num_intent = num_intent
        self.num_slot = num_slot
        self.use_freq_exit = getattr(args, "use_freq_exit", False)
        cfg = AutoConfig.from_pretrained(args.model_name_or_path)
        self.num_layers = cfg.num_hidden_layers
        self.wordrep = DecoderWordRep(args)
        floor = float(getattr(args, "min_unfrozen_ratio_floor", 0.5))
        ratio = max(floor, min(1.0, getattr(args, "unfrozen_ratio", 0.5)))
        self.unfreeze_position = getattr(args, "unfreeze_position", "front")
        if hasattr(self.wordrep.base_model, "layers"):
            total_bb_layers = len(self.wordrep.base_model.layers)
        else:
            total_bb_layers = self.num_layers
        self._total_bb_layers = total_bb_layers
        self.num_unfrozen_layers = max(0 if ratio == 0.0 else 1, round(total_bb_layers * ratio))
        self._apply_partial_freeze()
        intent_hidden = getattr(args, "intent_head_hidden", 0)
        slot_hidden = getattr(args, "slot_head_hidden", 0)
        self.exit_intent_heads = nn.ModuleList(
            [
                ExitHead(cfg.hidden_size, num_intent, args.dropout_rate, hidden_dim=intent_hidden)
                for _ in range(self.num_layers)
            ]
        )
        self.exit_slot_heads = nn.ModuleList(
            [
                ExitHead(cfg.hidden_size, num_slot, args.dropout_rate, hidden_dim=slot_hidden)
                for _ in range(self.num_layers)
            ]
        )
        _bdt = next(self.wordrep.base_model.parameters()).dtype
        for _cn, _cm in self.named_children():
            if _cn != "wordrep":
                _cm.to(dtype=_bdt)
        half = math.ceil(self.num_layers / 2)
        raw_mel = getattr(args, "min_exit_layer", None)
        if raw_mel is None:
            self.min_exit_layer = half
        elif raw_mel < half:
            logger.warning(
                "--min_exit_layer=%d is below 50%% of depth (num_layers=%d -> floor=%d). Clamping to %d: this architecture never exits before the halfway point.",
                raw_mel,
                self.num_layers,
                half,
                half,
            )
            self.min_exit_layer = half
        else:
            self.min_exit_layer = raw_mel
        self.patience = getattr(args, "ee_patience", 3)
        self.tau_intent = getattr(args, "tau_intent", 0.05)
        self.tau_slot = getattr(args, "tau_slot", 0.1)
        self.intent_margin = getattr(args, "intent_exit_margin", 0.15)
        self.patience_decay = getattr(args, "ee_patience_decay", 0.0)
        self.patience_min = getattr(args, "ee_patience_min", 1)
        self.exit_logit_smoothing = getattr(args, "exit_logit_smoothing", True)
        self.require_joint_stability = getattr(args, "require_joint_stability", True)
        n_trainable = sum((p.numel() for p in self.parameters() if p.requires_grad))
        n_total = sum((p.numel() for p in self.parameters()))
        logger.info(
            "Model: L=%d  min_exit=%d (floor=%d)  patience=%d (decay=%.2f min=%d)  tau_intent=%.4f  tau_slot=%.4f  freq_adaptive_exit=%s  exit_logit_smoothing=%s  unfrozen_backbone_layers=%d/%d  trainable_params=%d/%d (%.2f%%)",
            self.num_layers,
            self.min_exit_layer,
            half,
            self.patience,
            self.patience_decay,
            self.patience_min,
            self.tau_intent,
            self.tau_slot,
            self.use_freq_exit,
            self.exit_logit_smoothing,
            self.num_unfrozen_layers,
            self.num_layers,
            n_trainable,
            n_total,
            100.0 * n_trainable / max(n_total, 1),
        )

    def _select_unfrozen_layer_indices(self, total: int) -> Set[int]:
        n = self.num_unfrozen_layers
        pos = self.unfreeze_position
        q = max(1, math.ceil(total / 4))
        early_idx = set(range(0, q))
        middle_idx = set(range(q, min(2 * q, total)))
        late_idx = set(range(max(total - q, 0), total))
        if pos == "front":
            return set(range(min(n, total)))
        elif pos == "none":
            return set()
        elif pos == "all":
            return set(range(total))
        elif pos == "early":
            return early_idx
        elif pos == "middle":
            return middle_idx
        elif pos == "late":
            return late_idx
        elif pos == "early+middle":
            return early_idx | middle_idx
        elif pos == "middle+late":
            return middle_idx | late_idx
        elif pos == "early+late":
            return early_idx | late_idx
        else:
            logger.warning("Unknown unfreeze_position=%r; falling back to 'front'.", pos)
            return set(range(min(n, total)))

    def _apply_partial_freeze(self):
        bm = self.wordrep.base_model
        for p in bm.parameters():
            p.requires_grad_(False)
        if hasattr(bm, "layers"):
            total = len(bm.layers)
            unfrozen_idx = self._select_unfrozen_layer_indices(total)
            self.unfrozen_layer_indices = sorted(unfrozen_idx)
            for i, layer in enumerate(bm.layers):
                if i in unfrozen_idx:
                    for p in layer.parameters():
                        p.requires_grad_(True)
            logger.info(
                "Partial backbone unfreeze [position=%s]: layers=%s TRAINABLE (%d/%d), rest FROZEN (embeddings frozen too). unfrozen_ratio=%.2f.",
                self.unfreeze_position,
                self.unfrozen_layer_indices,
                len(unfrozen_idx),
                total,
                getattr(self.args, "unfrozen_ratio", 0.5),
            )
        else:
            logger.warning(
                "Backbone does not expose .layers -- cannot selectively unfreeze layers. Falling back to a FULLY FROZEN backbone (num_unfrozen_layers=0)."
            )
            self.num_unfrozen_layers = 0
            self.unfrozen_layer_indices = []
        bm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        bm = self.wordrep.base_model
        unfrozen_set = set(getattr(self, "unfrozen_layer_indices", []))
        if mode and self.num_unfrozen_layers > 0 and hasattr(bm, "layers"):
            bm.train(True)
            for i, layer in enumerate(bm.layers):
                layer.train(i in unfrozen_set)
        else:
            bm.eval()
        return self

    def forward(
        self,
        input_ids,
        attention_mask,
        words_lengths,
        word_attention_mask,
        return_layer_probes=False,
    ):
        device = input_ids.device
        need_grad = self.num_unfrozen_layers > 0 and torch.is_grad_enabled()
        if need_grad:
            out = self.wordrep.base_model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
        else:
            with torch.no_grad():
                out = self.wordrep.base_model(
                    input_ids, attention_mask=attention_mask, output_hidden_states=True
                )
        hs = out.hidden_states
        align = _build_align(input_ids, words_lengths, device).to(dtype=hs[-1].dtype)
        l_int, l_slot = ([], [])
        for l in range(self.num_layers):
            h = hs[l + 1]
            cls_l = self.wordrep.pooling(h, attention_mask)
            word_h_l = torch.bmm(align, h)
            l_int.append(self.exit_intent_heads[l](cls_l))
            l_slot.append(self.exit_slot_heads[l](word_h_l))
        if not return_layer_probes:
            return (l_int[-1], l_slot[-1])
        return (l_int, l_slot)

    def _true_layer_iter(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Generator[Tuple[int, torch.Tensor], None, None]:
        bm = self.wordrep.base_model
        device = input_ids.device
        B, T = input_ids.shape
        if not (hasattr(bm, "embed_tokens") and hasattr(bm, "layers")):
            raise RuntimeError(
                "Backbone does not expose .embed_tokens / .layers. True layer-by-layer early exit is unsupported for this model."
            )
        h = bm.embed_tokens(input_ids)
        position_ids = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        cache_position = torch.arange(T, device=device)
        if hasattr(bm, "rotary_emb"):
            position_embeddings = bm.rotary_emb(h, position_ids)
        else:
            position_embeddings = None
        causal_mask: Optional[torch.Tensor] = None
        if hasattr(bm, "_update_causal_mask"):
            for _kw in (
                dict(past_key_values=None, output_attentions=False),
                dict(past_key_values=None),
                {},
            ):
                try:
                    causal_mask = bm._update_causal_mask(attention_mask, h, cache_position, **_kw)
                    break
                except TypeError:
                    continue
        if causal_mask is None and (not hasattr(bm, "_update_causal_mask")):
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
                    f"Layer {l} output has unexpected shape {raw.shape}; expected 3-D (B={B}, T={T}, d)."
                )
            if raw.shape[0] == T and raw.shape[1] == B and (T != B):
                logger.warning(
                    "Layer %d output appears transposed (%s); correcting to (B, T, d).",
                    l,
                    tuple(raw.shape),
                )
                raw = raw.transpose(0, 1).contiguous()
            h = raw
            yield (l, h)

    @staticmethod
    def _discretize_intent(
        ip_l: torch.Tensor, is_multi_label: bool, intent_threshold: float
    ) -> torch.Tensor:
        if is_multi_label:
            return ip_l >= intent_threshold
        idx = ip_l.argmax(dim=-1)
        onehot = torch.zeros_like(ip_l, dtype=torch.bool)
        onehot.scatter_(1, idx.unsqueeze(1), True)
        return onehot

    @torch.no_grad()
    def forward_with_early_exit(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        words_lengths: torch.Tensor,
        word_attention_mask: torch.Tensor,
        freq_scores: Optional[torch.Tensor] = None,
        intent_threshold: float = 0.5,
        is_multi_label: bool = True,
    ) -> Tuple:
        B = input_ids.size(0)
        device = input_ids.device
        dummy = next(self.wordrep.base_model.parameters())
        align = _build_align(input_ids, words_lengths, device).to(dtype=dummy.dtype)
        wam_f = word_attention_mask.float()
        wam_len = wam_f.sum(dim=1).clamp(min=1.0)
        if self.use_freq_exit and freq_scores is not None:
            fs = freq_scores.float().to(device).clamp(0.0, 1.0)
            span = float(self.num_layers - 1 - self.min_exit_layer)
            per_min = (
                (self.min_exit_layer + fs * span)
                .long()
                .clamp(self.min_exit_layer, self.num_layers - 1)
            )
        else:
            per_min = torch.full((B,), self.min_exit_layer, dtype=torch.long, device=device)
        pat_cnt = torch.zeros(B, dtype=torch.long, device=device)
        exited = torch.zeros(B, dtype=torch.bool, device=device)
        exit_lyr = torch.full((B,), self.num_layers - 1, dtype=torch.long, device=device)
        exit_int: List[Optional[torch.Tensor]] = [None] * B
        exit_slot: List[Optional[torch.Tensor]] = [None] * B
        prev_ip: Optional[torch.Tensor] = None
        prev_intent_pred: Optional[torch.Tensor] = None
        prev_slot_pred: Optional[torch.Tensor] = None
        prev_int_logits: Optional[torch.Tensor] = None
        prev_slot_logits: Optional[torch.Tensor] = None
        last_int_logits: Optional[torch.Tensor] = None
        last_slot_logits: Optional[torch.Tensor] = None
        for l, h in self._true_layer_iter(input_ids, attention_mask):
            cls_l = self.wordrep.pooling(h, attention_mask)
            word_h_l = torch.bmm(align, h)
            int_logits_l = self.exit_intent_heads[l](cls_l)
            slot_logits_l = self.exit_slot_heads[l](word_h_l)
            last_int_logits, last_slot_logits = (int_logits_l, slot_logits_l)
            ip_l = torch.sigmoid(int_logits_l)
            slot_pred_l = slot_logits_l.argmax(dim=-1)
            intent_pred_l = self._discretize_intent(ip_l, is_multi_label, intent_threshold)
            if prev_intent_pred is not None:
                eligible = ~exited & (l >= per_min)
                intent_agree = (intent_pred_l == prev_intent_pred).all(dim=-1)
                margin_ok = (ip_l - intent_threshold).abs().amax(dim=-1) >= self.intent_margin
                intent_stable = intent_agree & margin_ok
                disagree = (slot_pred_l != prev_slot_pred).float() * wam_f
                frac_disagree = disagree.sum(dim=1) / wam_len
                slot_stable = frac_disagree <= self.tau_slot
                joint_stable = (
                    intent_stable & slot_stable if self.require_joint_stability else intent_stable
                )
                stable = eligible & joint_stable
                unstable = eligible & ~joint_stable
                pat_cnt = (pat_cnt + stable.long()) * (~unstable).long()
                depth_since_min = (l - per_min).clamp(min=0).float()
                required_patience = (self.patience - self.patience_decay * depth_since_min).clamp(
                    min=float(self.patience_min)
                )
                new_exits = eligible & (pat_cnt.float() >= required_patience)
                if new_exits.any():
                    idx = new_exits.nonzero(as_tuple=True)[0].tolist()
                    for i in idx:
                        exit_lyr[i] = l
                        if self.exit_logit_smoothing and prev_int_logits is not None:
                            exit_int[i] = (
                                (0.5 * (int_logits_l[i] + prev_int_logits[i])).detach().clone()
                            )
                            exit_slot[i] = (
                                (0.5 * (slot_logits_l[i] + prev_slot_logits[i])).detach().clone()
                            )
                        else:
                            exit_int[i] = int_logits_l[i].detach().clone()
                            exit_slot[i] = slot_logits_l[i].detach().clone()
                    exited = exited | new_exits
            prev_ip = ip_l.detach()
            prev_intent_pred = intent_pred_l.detach()
            prev_slot_pred = slot_pred_l.detach()
            prev_int_logits = int_logits_l.detach()
            prev_slot_logits = slot_logits_l.detach()
            if exited.all():
                break
        assert last_int_logits is not None, "No layers were iterated — backbone has no .layers?"
        for i in range(B):
            if exit_int[i] is None:
                exit_int[i] = last_int_logits[i].detach().clone()
                exit_slot[i] = last_slot_logits[i].detach().clone()
        final_int = torch.stack(exit_int, dim=0)
        final_slot = torch.stack(exit_slot, dim=0)
        return ((final_int, final_slot), exit_lyr)


class FrozenFeatureCache:

    def __init__(self, model: "JointModelWithEarlyExit", device: str):
        self.model = model
        self.device = device
        self.cls_cache: Optional[torch.Tensor] = None
        self.word_cache: Optional[torch.Tensor] = None
        self.built = False

    def estimate_bytes(self, n: int, max_words: int) -> int:
        L = self.model.num_layers
        D = next(self.model.wordrep.base_model.parameters()).shape[-1]
        return 2 * n * L * D * (1 + max_words)

    @torch.no_grad()
    def build(self, dataset, pad_id: int, batch_size: int, max_gb: float = 6.0) -> bool:
        n = len(dataset)
        max_words = dataset[0][2].shape[0] if n > 0 else 0
        est_bytes = self.estimate_bytes(n, max_words)
        est_gb = est_bytes / 1024**3
        if est_gb > max_gb:
            logger.warning(
                "FrozenFeatureCache: estimated %.2f GB exceeds --cache_max_gb=%.2f GB for %d examples. Skipping cache -- falling back to the normal (recomputed-every-epoch) training path. Raise --cache_max_gb, shrink --max_seq_length, or use a smaller train split to enable it.",
                est_gb,
                max_gb,
                n,
            )
            return False
        logger.info(
            "FrozenFeatureCache: building for %d examples (~%.2f GB, fp16 CPU) ...", n, est_gb
        )
        t0 = time.perf_counter()
        L = self.model.num_layers
        D = next(self.model.wordrep.base_model.parameters()).shape[-1]
        self.cls_cache = torch.empty((n, L, D), dtype=torch.float16)
        self.word_cache = torch.empty((n, L, max_words, D), dtype=torch.float16)
        dl = DataLoader(
            dataset,
            sampler=SequentialSampler(dataset),
            num_workers=2,
            batch_size=batch_size,
            collate_fn=lambda x: collate_fn(x, pad_id),
        )
        self.model.eval()
        write_ptr = 0
        for batch in dl:
            bsz = batch[0].size(0)
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            words_lengths = batch[2].to(self.device)
            out = self.model.wordrep.base_model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
            hs = out.hidden_states
            align = _build_align(input_ids, words_lengths, self.device).to(dtype=hs[-1].dtype)
            for l in range(L):
                h = hs[l + 1]
                cls_l = self.model.wordrep.pooling(h, attention_mask)
                word_h_l = torch.bmm(align, h)
                self.cls_cache[write_ptr : write_ptr + bsz, l] = cls_l.detach().to(
                    "cpu", torch.float16
                )
                self.word_cache[write_ptr : write_ptr + bsz, l] = word_h_l.detach().to(
                    "cpu", torch.float16
                )
            write_ptr += bsz
        self.built = True
        logger.info("FrozenFeatureCache: built in %.1fs.", time.perf_counter() - t0)
        return True


class CachedFeatureDataset(Dataset):

    def __init__(self, base_dataset, cache: FrozenFeatureCache):
        self.base = base_dataset
        self.cache = cache

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        _iids, _amask, _wlen, wattn, ilbl, slbl, fscr = self.base[idx]
        return (self.cache.cls_cache[idx], self.cache.word_cache[idx], wattn, ilbl, slbl, fscr)


def collate_fn_cached(batch):
    cls_l, word_l, wattn, ilbl, slbl, fscr = zip(*batch)
    return (
        torch.stack(cls_l),
        torch.stack(word_l),
        torch.stack(wattn),
        torch.stack(ilbl),
        torch.stack(slbl),
        torch.tensor(fscr, dtype=torch.float),
    )


class EarlyExitTrainer:

    def __init__(
        self,
        args,
        tokenizer,
        train_ds,
        dev_ds,
        test_ds,
        intent_label_set,
        slot_label_set,
        intent_pos_weight: Optional[torch.Tensor] = None,
    ):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.trainer_state = TrainerState()
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "dev_loss": [],
            "dev_tuning_metric": [],
        }
        self.last_exit_layers: List[int] = []
        self.last_freq_scores: List[float] = []
        self.last_intent_true: Optional[np.ndarray] = None
        self.last_intent_pred: Optional[np.ndarray] = None
        self.intent_label_set = intent_label_set
        self.slot_label_set = slot_label_set
        self.train_ds = train_ds
        self.dev_ds = dev_ds
        self.test_ds = test_ds
        self.model = JointModelWithEarlyExit(args, len(intent_label_set), len(slot_label_set)).to(
            self.device
        )
        self.intent_pos_weight = (
            intent_pos_weight.to(self.device) if intent_pos_weight is not None else None
        )
        self.intent_loss_fn = getattr(args, "intent_loss_fn", "asl")
        self.asl_gamma_neg = getattr(args, "asl_gamma_neg", 4.0)
        self.asl_gamma_pos = getattr(args, "asl_gamma_pos", 0.0)
        self.asl_clip = getattr(args, "asl_clip", 0.05)
        if self.intent_loss_fn == "asl":
            logger.info(
                "Intent loss = Asymmetric Loss (gamma_neg=%.2f gamma_pos=%.2f clip=%.3f); pos_weight is still computed/logged above but NOT applied under ASL (ASL's own focusing + probability-shift already correct for imbalance). Pass --intent_loss_fn bce to use plain BCE+pos_weight instead.",
                self.asl_gamma_neg,
                self.asl_gamma_pos,
                self.asl_clip,
            )
        self.intent_threshold = getattr(args, "intent_threshold_init", 0.5)
        self.is_multi_label = self._infer_is_multi_label(
            train_ds if train_ds is not None else dev_ds
        )
        logger.info(
            "Detected task modality: is_multi_label=%s (used by the PABEE exit criterion; recomputed from data, not guessed).",
            self.is_multi_label,
        )
        self.train_probe_ds = None
        if train_ds is not None:
            n = min(getattr(args, "train_probe_size", 1000), len(train_ds))
            g = torch.Generator().manual_seed(42)
            idx = torch.randperm(len(train_ds), generator=g)[:n].tolist()
            self.train_probe_ds = torch.utils.data.Subset(train_ds, idx)
        if _wandb_active:
            wb.watch(
                self.model,
                log="all",
                log_freq=getattr(args, "wandb_watch_freq", 100),
                log_graph=False,
            )

    @staticmethod
    def _infer_is_multi_label(ds, sample_size: int = 500) -> bool:
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
            ds,
            sampler=s,
            num_workers=4 if shuffle else 2,
            batch_size=b,
            collate_fn=lambda x: collate_fn(x, self.tokenizer.pad_token_id),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=shuffle,
        )

    def _intent_loss(self, y_hat, y_true):
        return intent_loss_func(
            y_hat,
            y_true,
            pos_weight=self.intent_pos_weight,
            loss_fn=self.intent_loss_fn,
            asl_gamma_neg=self.asl_gamma_neg,
            asl_gamma_pos=self.asl_gamma_pos,
            asl_clip=self.asl_clip,
        )

    def compute_loss(
        self,
        model,
        inputs,
        slot_labels,
        intent_labels,
        word_attention_mask,
        freq_scores: Optional[torch.Tensor] = None,
    ):
        l_int, l_slot = model(**inputs, return_layer_probes=True)
        L = len(l_int)
        use_freq_loss = getattr(self.args, "use_freq_exit", False) and freq_scores is not None
        mean_rarity = float(freq_scores.mean().item()) if use_freq_loss else 0.5
        total = torch.tensor(0.0, device=self.device)
        intent_diag = torch.tensor(0.0, device=self.device)
        slot_diag = torch.tensor(0.0, device=self.device)
        for l in range(L):
            w = mean_rarity * (l + 1) / L + (1.0 - mean_rarity) * (L - l) / L
            intent_l = self._intent_loss(l_int[l], intent_labels.float())
            s_out, s_lbl = get_useful_ones(l_slot[l], slot_labels, word_attention_mask)
            if s_lbl.numel() > 0:
                slot_l = F.cross_entropy(s_out, s_lbl)
            else:
                slot_l = l_slot[l].sum() * 0.0
            total = total + w * (
                self.args.loss_coef_intent * intent_l + self.args.loss_coef_slot * slot_l
            )
            intent_diag = intent_diag + intent_l.detach()
            slot_diag = slot_diag + (slot_l.detach() if torch.is_tensor(slot_l) else slot_l)
        total = total / L
        intent_diag = intent_diag / L
        slot_diag = slot_diag / L
        final_intent_l = self._intent_loss(l_int[-1], intent_labels.float()).detach()
        fs_out, fs_lbl = get_useful_ones(l_slot[-1], slot_labels, word_attention_mask)
        final_slot_l = (
            F.cross_entropy(fs_out, fs_lbl)
            if fs_lbl.numel() > 0
            else l_slot[-1].sum().detach() * 0.0
        )
        final_layer_loss = (
            self.args.loss_coef_intent * final_intent_l + self.args.loss_coef_slot * final_slot_l
        )
        return (total, l_int[-1], l_slot[-1], intent_diag, slot_diag, final_layer_loss)

    def compute_loss_from_cache(
        self,
        cls_stack,
        word_stack,
        slot_labels,
        intent_labels,
        word_attention_mask,
        freq_scores: Optional[torch.Tensor] = None,
    ):
        model = self.model
        L = model.num_layers
        head_dtype = next(model.exit_intent_heads[0].parameters()).dtype
        use_freq_loss = getattr(self.args, "use_freq_exit", False) and freq_scores is not None
        mean_rarity = float(freq_scores.mean().item()) if use_freq_loss else 0.5
        total = torch.tensor(0.0, device=self.device)
        intent_diag = torch.tensor(0.0, device=self.device)
        slot_diag = torch.tensor(0.0, device=self.device)
        l_int_last, l_slot_last = (None, None)
        for l in range(L):
            cls_l = cls_stack[:, l].to(head_dtype)
            word_h_l = word_stack[:, l].to(head_dtype)
            int_logits_l = model.exit_intent_heads[l](cls_l)
            slot_logits_l = model.exit_slot_heads[l](word_h_l)
            l_int_last, l_slot_last = (int_logits_l, slot_logits_l)
            w = mean_rarity * (l + 1) / L + (1.0 - mean_rarity) * (L - l) / L
            intent_l = self._intent_loss(int_logits_l, intent_labels.float())
            s_out, s_lbl = get_useful_ones(slot_logits_l, slot_labels, word_attention_mask)
            slot_l = (
                F.cross_entropy(s_out, s_lbl) if s_lbl.numel() > 0 else slot_logits_l.sum() * 0.0
            )
            total = total + w * (
                self.args.loss_coef_intent * intent_l + self.args.loss_coef_slot * slot_l
            )
            intent_diag = intent_diag + intent_l.detach()
            slot_diag = slot_diag + (slot_l.detach() if torch.is_tensor(slot_l) else slot_l)
        total = total / L
        intent_diag = intent_diag / L
        slot_diag = slot_diag / L
        final_intent_l = self._intent_loss(l_int_last, intent_labels.float()).detach()
        fs_out, fs_lbl = get_useful_ones(l_slot_last, slot_labels, word_attention_mask)
        final_slot_l = (
            F.cross_entropy(fs_out, fs_lbl)
            if fs_lbl.numel() > 0
            else l_slot_last.sum().detach() * 0.0
        )
        final_layer_loss = (
            self.args.loss_coef_intent * final_intent_l + self.args.loss_coef_slot * final_slot_l
        )
        return (total, l_int_last, l_slot_last, intent_diag, slot_diag, final_layer_loss)

    def _build_optimizer(self):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        backbone_prefix = "wordrep.base_model."
        backbone_params = [
            (n, p)
            for n, p in self.model.named_parameters()
            if p.requires_grad and n.startswith(backbone_prefix)
        ]
        head_params = [
            (n, p)
            for n, p in self.model.named_parameters()
            if p.requires_grad and (not n.startswith(backbone_prefix))
        ]
        n_bb = sum((p.numel() for _, p in backbone_params))
        n_head = sum((p.numel() for _, p in head_params))
        n_total = sum((p.numel() for p in self.model.parameters()))
        backbone_lr = getattr(self.args, "backbone_learning_rate", None)
        if not backbone_lr:
            backbone_lr = self.args.learning_rate * 0.1
        logger.info(
            "Optimiser: heads=%d params (lr=%.2e), unfrozen backbone=%d params (lr=%.2e), frozen=%d params. Total trainable %d/%d (%.2f%%).",
            n_head,
            self.args.learning_rate,
            n_bb,
            backbone_lr,
            n_total - n_bb - n_head,
            n_bb + n_head,
            n_total,
            100.0 * (n_bb + n_head) / max(n_total, 1),
        )
        groups = []
        if head_params:
            groups.append(
                {
                    "params": [p for n, p in head_params if not any((x in n for x in no_decay))],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                }
            )
            groups.append(
                {
                    "params": [p for n, p in head_params if any((x in n for x in no_decay))],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                }
            )
        if backbone_params:
            groups.append(
                {
                    "params": [
                        p for n, p in backbone_params if not any((x in n for x in no_decay))
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": backbone_lr,
                }
            )
            groups.append(
                {
                    "params": [p for n, p in backbone_params if any((x in n for x in no_decay))],
                    "weight_decay": 0.0,
                    "lr": backbone_lr,
                }
            )
        return AdamW(groups, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

    def train(self):
        global wb, _wandb_active
        use_cache = False
        if getattr(self.args, "cache_frozen_features", False):
            if getattr(self.model, "num_unfrozen_layers", 0) > 0:
                logger.warning(
                    "--cache_frozen_features requested but %d backbone layers are TRAINABLE (unfrozen_ratio=%.2f). Cached per-layer features assume a fully frozen, deterministic backbone (see FrozenFeatureCache docstring) -- with trainable layers the cached features go stale after the very first optimizer step, which would silently train the heads against outdated features. Ignoring --cache_frozen_features and using the normal recompute-every-step path instead.",
                    self.model.num_unfrozen_layers,
                    getattr(self.args, "unfrozen_ratio", 0.5),
                )
            else:
                cache = FrozenFeatureCache(self.model, self.device)
                use_cache = cache.build(
                    self.train_ds,
                    self.tokenizer.pad_token_id,
                    batch_size=self.args.eval_batch_size,
                    max_gb=getattr(self.args, "cache_max_gb", 6.0),
                )
        if use_cache:
            cached_ds = CachedFeatureDataset(self.train_ds, cache)
            dl = DataLoader(
                cached_ds,
                sampler=RandomSampler(cached_ds),
                batch_size=self.args.train_batch_size,
                collate_fn=collate_fn_cached,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            dl = self._dl(self.train_ds, True)
        steps = len(dl) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        opt = self._build_optimizer()
        sched = get_linear_schedule_with_warmup(
            opt, int(self.args.warmup_proportion * steps), steps
        )
        use_amp = getattr(self.args, "use_amp", False) and torch.cuda.is_available()
        logger.info(
            "Training: steps=%d  device=%s  L=%d  min_exit=%d  patience=%d  tau_intent=%.4f  tau_slot=%.4f  AMP=%s  freq_adaptive=%s  intent_loss_fn=%s  cache_frozen_features=%s(used=%s)",
            steps,
            self.device,
            self.model.num_layers,
            self.model.min_exit_layer,
            self.model.patience,
            self.model.tau_intent,
            self.model.tau_slot,
            use_amp,
            getattr(self.args, "use_freq_exit", False),
            self.intent_loss_fn,
            getattr(self.args, "cache_frozen_features", False),
            use_cache,
        )
        if _wandb_active:
            wb.log(
                {
                    "dataset/train_size": len(self.train_ds),
                    "dataset/dev_size": len(self.dev_ds),
                    "dataset/test_size": len(self.test_ds),
                    "model/num_layers": self.model.num_layers,
                    "model/num_intent": self.model.num_intent,
                    "model/num_slot": self.model.num_slot,
                    "model/min_exit_layer": self.model.min_exit_layer,
                },
                step=0,
            )
        es = EarlyStopping(self.args.early_stopping, verbose=True)
        self.model.zero_grad()
        gs = 0
        total_samples_seen = 0
        t_epoch_start = time.perf_counter()
        for epoch in trange(self.args.num_train_epochs):
            self.model.train()
            ep_loss = 0.0
            ep_steps = 0
            ep_intent_loss = 0.0
            ep_slot_loss = 0.0
            ep_final_layer_loss = 0.0
            t_step = time.perf_counter()
            for step, batch in enumerate(dl):
                batch_size = batch[0].size(0)
                opt.zero_grad()
                if use_cache:
                    cls_stack = batch[0].to(self.device)
                    word_stack = batch[1].to(self.device)
                    word_attn = batch[2].to(self.device)
                    intent_labels = batch[3].to(self.device)
                    slot_labels = batch[4].to(self.device)
                    freq_scores = batch[5].to(self.device)
                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            loss, _, _, intent_l_diag, slot_l_diag, final_layer_loss = (
                                self.compute_loss_from_cache(
                                    cls_stack,
                                    word_stack,
                                    slot_labels,
                                    intent_labels,
                                    word_attn,
                                    freq_scores=freq_scores,
                                )
                            )
                    else:
                        loss, _, _, intent_l_diag, slot_l_diag, final_layer_loss = (
                            self.compute_loss_from_cache(
                                cls_stack,
                                word_stack,
                                slot_labels,
                                intent_labels,
                                word_attn,
                                freq_scores=freq_scores,
                            )
                        )
                else:
                    freq_scores = batch[6].to(self.device)
                    inputs = {
                        "input_ids": batch[0].to(self.device),
                        "attention_mask": batch[1].to(self.device),
                        "words_lengths": batch[2].to(self.device),
                        "word_attention_mask": batch[3].to(self.device),
                    }
                    slot_labels = batch[5].to(self.device)
                    intent_labels = batch[4].to(self.device)
                    word_attn = batch[3].to(self.device)
                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            loss, _, _, intent_l_diag, slot_l_diag, final_layer_loss = (
                                self.compute_loss(
                                    self.model,
                                    inputs,
                                    slot_labels,
                                    intent_labels,
                                    word_attn,
                                    freq_scores=freq_scores,
                                )
                            )
                    else:
                        loss, _, _, intent_l_diag, slot_l_diag, final_layer_loss = (
                            self.compute_loss(
                                self.model,
                                inputs,
                                slot_labels,
                                intent_labels,
                                word_attn,
                                freq_scores=freq_scores,
                            )
                        )
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                if not torch.isfinite(loss):
                    logger.warning("Non-finite loss epoch=%d step=%d. Skipping.", epoch, step)
                    opt.zero_grad(set_to_none=True)
                    self.model.zero_grad(set_to_none=True)
                    continue
                ep_loss += loss.item()
                ep_steps += 1
                ep_intent_loss += intent_l_diag.item()
                ep_slot_loss += slot_l_diag.item()
                ep_final_layer_loss += final_layer_loss.item()
                loss.backward()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )
                    if not torch.isfinite(grad_norm):
                        logger.warning(
                            "Non-finite grad norm epoch=%d step=%d. Skipping.", epoch, step
                        )
                        opt.zero_grad(set_to_none=True)
                        self.model.zero_grad(set_to_none=True)
                        continue
                    opt.step()
                    sched.step()
                    self.model.zero_grad(set_to_none=True)
                    gs += 1
                    total_samples_seen += batch_size
                    self.trainer_state.epoch = epoch
                    self.trainer_state.global_step = gs
                    self.trainer_state.max_steps = steps
                    self.trainer_state.loss = ep_loss / max(ep_steps, 1)
                    if _wandb_active:
                        t_now = time.perf_counter()
                        elapsed = max(t_now - t_step, 1e-06)
                        t_step = t_now
                        wb.log(
                            {
                                "train/loss": loss.item() * self.args.gradient_accumulation_steps,
                                "train/loss_smoothed": ep_loss / max(ep_steps, 1),
                                "train/intent_loss": intent_l_diag.item(),
                                "train/slot_loss": slot_l_diag.item(),
                                "train/intent_loss_smoothed": ep_intent_loss / max(ep_steps, 1),
                                "train/slot_loss_smoothed": ep_slot_loss / max(ep_steps, 1),
                                "train/final_layer_loss": final_layer_loss.item(),
                                "train/final_layer_loss_smoothed": ep_final_layer_loss
                                / max(ep_steps, 1),
                                "train/learning_rate": sched.get_last_lr()[0],
                                "train/grad_norm": (
                                    grad_norm.item()
                                    if torch.is_tensor(grad_norm)
                                    else float(grad_norm)
                                ),
                                "perf/samples_per_sec": batch_size / elapsed,
                                "perf/total_samples_seen": total_samples_seen,
                                "train/epoch": epoch + (step + 1) / len(dl),
                                "train/batch_mean_rarity": freq_scores.mean().item(),
                                **_gpu_mem_stats(self.device),
                            },
                            step=gs,
                        )
                if (step + 1) % self.args.logging_steps == 0:
                    logger.info(self.trainer_state.to_string())
            epoch_loss = ep_loss / max(ep_steps, 1)
            epoch_intent_loss = ep_intent_loss / max(ep_steps, 1)
            epoch_slot_loss = ep_slot_loss / max(ep_steps, 1)
            epoch_final_layer_loss = ep_final_layer_loss / max(ep_steps, 1)
            epoch_time = time.perf_counter() - t_epoch_start
            t_epoch_start = time.perf_counter()
            logger.info(
                "Epoch %d done: total_loss(all-layer avg)=%.5f  final_layer_loss=%.5f  intent_loss=%.5f  slot_loss=%.5f",
                epoch,
                epoch_loss,
                epoch_final_layer_loss,
                epoch_intent_loss,
                epoch_slot_loss,
            )
            if _wandb_active:
                wb.log(
                    {
                        "epoch/train_loss": epoch_loss,
                        "epoch/train_final_layer_loss": epoch_final_layer_loss,
                        "epoch/train_intent_loss": epoch_intent_loss,
                        "epoch/train_slot_loss": epoch_slot_loss,
                        "epoch/epoch_time_sec": epoch_time,
                        "epoch/epoch": epoch,
                    },
                    step=gs,
                )
            results = self.evaluate("dev", global_step=gs, epoch=epoch)
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(epoch_loss)
            self.history["dev_loss"].append(results.get("loss", float("nan")))
            self.history["dev_tuning_metric"].append(
                results.get(self.args.tuning_metric, float("nan"))
            )
            if self.train_probe_ds is not None and epoch % max(1, self.args.train_probe_every) == 0:
                tp_results = self.evaluate(
                    "train_probe", global_step=gs, epoch=epoch, ds_override=self.train_probe_ds
                )
                logger.info(
                    "  Train-probe (dev-comparable, exit-based): loss=%.5f intent_acc=%.4f intent_micro_f1=%.4f slot_f1=%.4f mean_intent_slot=%.4f mean_f1=%.4f",
                    tp_results["loss"],
                    tp_results["intent_acc"],
                    tp_results["intent_micro_f1"],
                    tp_results["slot_f1"],
                    tp_results["mean_intent_slot"],
                    tp_results["mean_f1"],
                )
            es(results[self.args.tuning_metric], self.args)
            if es.counter == 0:
                self.save_model()
            if es.early_stop:
                logger.info("Early stopping.")
                break
        wb.finish()
        wb = _WandbDummy()
        _wandb_active = False

    def evaluate(
        self,
        mode="dev",
        global_step: int = 0,
        epoch: int = 0,
        ds_override=None,
        log_wandb: bool = True,
        quiet: bool = False,
    ):
        ds = (
            ds_override
            if ds_override is not None
            else {"dev": self.dev_ds, "test": self.test_ds}.get(mode)
        )
        if ds is None:
            raise ValueError(f"mode {mode!r} needs ds_override or must be 'dev'/'test'.")
        if not quiet:
            logger.info("Eval [%s] %d samples", mode, len(ds))
        dl = self._dl(ds, False)
        self.model.eval()
        ev_loss = 0.0
        int_la, int_pa, slot_la, slot_pa, mask_a = ([], [], [], [], [])
        all_exit_lyrs: List[int] = []
        all_freq_scores: List[float] = []
        layer_exit_counts = defaultdict(int)
        t_eval_start = time.perf_counter()
        for batch in dl:
            freq_scores = batch[6].to(self.device)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0].to(self.device),
                    "attention_mask": batch[1].to(self.device),
                    "words_lengths": batch[2].to(self.device),
                    "word_attention_mask": batch[3].to(self.device),
                }
                il = batch[4].to(self.device)
                sl = batch[5].to(self.device)
                wam = batch[3].to(self.device)
                (final_i, final_s), exit_lyr_batch = self.model.forward_with_early_exit(
                    **inputs,
                    freq_scores=freq_scores,
                    intent_threshold=self.intent_threshold,
                    is_multi_label=self.is_multi_label,
                )
                for li in exit_lyr_batch.tolist():
                    all_exit_lyrs.append(li)
                    layer_exit_counts[li] += 1
                all_freq_scores.extend(freq_scores.cpu().tolist())
                s_out, s_lbl = get_useful_ones(final_s, sl, wam)
                slot_loss = (
                    F.cross_entropy(s_out, s_lbl) if s_lbl.numel() > 0 else final_s.sum() * 0.0
                )
                ev_loss += (
                    self.args.loss_coef_intent * self._intent_loss(final_i, il.float())
                    + self.args.loss_coef_slot * slot_loss
                ).item()
            int_la.append(il)
            int_pa.append(final_i)
            slot_la.append(sl)
            slot_pa.append(final_s)
            mask_a.append(wam)
        eval_time = time.perf_counter() - t_eval_start
        ev_loss /= len(dl)
        del dl
        results = {"loss": ev_loss}
        int_pa_cat, int_la_cat = (torch.cat(int_pa, 0), torch.cat(int_la, 0))
        if mode == "dev" and self.is_multi_label:
            probs_np = torch.sigmoid(int_pa_cat.detach().float().cpu()).numpy()
            labels_np = int_la_cat.detach().cpu().numpy().astype(int)
            best_t, best_f1 = search_best_intent_threshold(probs_np, labels_np)
            if not quiet:
                logger.info(
                    "Dev intent-threshold search: best_threshold=%.2f (dev micro-F1=%.4f), previous=%.2f",
                    best_t,
                    best_f1,
                    self.intent_threshold,
                )
            self.intent_threshold = best_t
        results.update(
            compute_metrics(
                self.args,
                int_pa_cat,
                int_la_cat,
                torch.cat(slot_pa, 0),
                torch.cat(slot_la, 0),
                torch.cat(mask_a, 0),
                self.slot_label_set,
                intent_threshold=self.intent_threshold,
            )
        )
        et = torch.tensor(all_exit_lyrs, dtype=torch.float)
        ml = self.model.num_layers - 1
        me = et.mean().item()
        se = et.std().item() if len(et) > 1 else 0.0
        results.update(
            {
                "mean_exit_layer": me,
                "std_exit_layer": se,
                "pct_full_pass": (et == ml).float().mean().item(),
                "layer_savings_pct": 1.0 - me / max(ml, 1),
            }
        )
        results["balanced_score"] = compute_balanced_score(results, self.args)
        if not quiet:
            for k in sorted(results):
                logger.info("  %-25s = %s", k, results[k])
            logger.info(
                "  Exit: mean=%.2f std=%.2f full_pass=%.1f%% savings=%.1f%%",
                me,
                se,
                results["pct_full_pass"] * 100,
                results["layer_savings_pct"] * 100,
            )
            logger.info("  Exit layer distribution:")
            for li in sorted(layer_exit_counts):
                pct = 100.0 * layer_exit_counts[li] / max(len(all_exit_lyrs), 1)
                logger.info("    layer %2d : %d samples (%.1f%%)", li, layer_exit_counts[li], pct)
        if _wandb_active and log_wandb:
            prefix = mode
            log_dict: Dict = {f"{prefix}/{k}": v for k, v in results.items()}
            log_dict[f"{prefix}/eval_time_sec"] = eval_time
            log_dict[f"{prefix}/samples_per_sec"] = len(ds) / max(eval_time, 1e-06)
            if all_exit_lyrs:
                log_dict[f"{prefix}/exit_layer_histogram"] = wb.Histogram(
                    all_exit_lyrs, num_bins=self.model.num_layers
                )
                table = wb.Table(columns=["layer", "sample_count", "pct"])
                for li in range(self.model.num_layers):
                    cnt = layer_exit_counts.get(li, 0)
                    table.add_data(li, cnt, 100.0 * cnt / max(len(all_exit_lyrs), 1))
                log_dict[f"{prefix}/exit_layer_table"] = table
            if all_freq_scores and getattr(self.args, "use_freq_exit", False):
                fs_arr = np.array(all_freq_scores)
                el_arr = np.array(all_exit_lyrs, dtype=np.float32)
                q_edges = np.quantile(fs_arr, [0.0, 0.25, 0.5, 0.75, 1.0])
                q_labels = ["Q1_frequent", "Q2", "Q3", "Q4_rare"]
                strat_table = wb.Table(
                    columns=[
                        "quartile",
                        "n_samples",
                        "rarity_mean",
                        "mean_exit_layer",
                        "pct_savings",
                    ]
                )
                for qi in range(4):
                    m = (fs_arr >= q_edges[qi]) & (fs_arr <= q_edges[qi + 1])
                    if not m.any():
                        continue
                    el_q = float(el_arr[m].mean())
                    sav_q = 1.0 - el_q / max(ml, 1)
                    strat_table.add_data(
                        q_labels[qi], int(m.sum()), float(fs_arr[m].mean()), el_q, sav_q
                    )
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
        self.last_exit_layers = list(all_exit_lyrs)
        self.last_freq_scores = list(all_freq_scores)
        try:
            il_cpu = int_la_cat.detach().cpu()
            if torch.all(il_cpu.sum(dim=1) == 1):
                self.last_intent_true = il_cpu.argmax(dim=1).numpy()
                self.last_intent_pred = int_pa_cat.detach().float().cpu().argmax(dim=1).numpy()
            else:
                self.last_intent_true, self.last_intent_pred = (None, None)
        except Exception:
            self.last_intent_true, self.last_intent_pred = (None, None)
        if not quiet:
            self._write(f"eval_{mode}_results.txt", results)
        return results

    def calibrate_exit_hparams(self, patience_grid=None, min_exit_grid=None):
        if patience_grid is None:
            base = self.model.patience
            patience_grid = sorted(set((max(1, v) for v in (base - 1, base, base + 1, base + 2))))
        if min_exit_grid is None:
            half = math.ceil(self.model.num_layers / 2)
            min_exit_grid = sorted(
                set(
                    (
                        v
                        for v in (half, half + 1, half + 2, self.model.num_layers - 1)
                        if half <= v <= self.model.num_layers - 1
                    )
                )
            )
        base_patience, base_min_exit = (self.model.patience, self.model.min_exit_layer)
        best = None
        logger.info(
            "Calibrating exit hyperparameters on DEV: patience in %s, min_exit_layer in %s",
            patience_grid,
            min_exit_grid,
        )
        for pat in patience_grid:
            for me in min_exit_grid:
                self.model.patience = pat
                self.model.min_exit_layer = me
                res = self.evaluate("dev", global_step=-1, epoch=-1, log_wandb=False, quiet=True)
                score, savings = (res[self.args.tuning_metric], res["layer_savings_pct"])
                logger.info(
                    "  patience=%d min_exit=%d -> %s=%.4f  layer_savings=%.1f%%",
                    pat,
                    me,
                    self.args.tuning_metric,
                    score,
                    savings * 100,
                )
                if best is None or score > best[0]:
                    best = (score, pat, me, savings)
        if best is None or best[0] < 0:
            logger.warning(
                "Exit calibration found nothing better than the current settings; keeping patience=%d min_exit_layer=%d.",
                base_patience,
                base_min_exit,
            )
            self.model.patience, self.model.min_exit_layer = (base_patience, base_min_exit)
            return {"patience": base_patience, "min_exit_layer": base_min_exit}
        self.model.patience, self.model.min_exit_layer = (best[1], best[2])
        logger.info(
            "Exit calibration selected: patience=%d min_exit_layer=%d (dev %s=%.4f, layer_savings=%.1f%%)",
            best[1],
            best[2],
            self.args.tuning_metric,
            best[0],
            best[3] * 100,
        )
        return {"patience": best[1], "min_exit_layer": best[2], "score": best[0]}

    def save_model(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        save_path = os.path.join(self.args.output_dir, "checkpoint.pth")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "intent_label_set": self.intent_label_set,
                "slot_label_set": self.slot_label_set,
            },
            save_path,
        )
        torch.save(self.args, os.path.join(self.args.output_dir, "training_args.bin"))
        self.trainer_state.save_to_json(os.path.join(self.args.output_dir, "trainer_state.json"))
        logger.info("Saved -> %s", save_path)

    def load_model(self):
        ckpt_path = os.path.join(self.args.output_dir, "checkpoint.pth")
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Loaded -> %s", ckpt_path)

    def _write(self, fname, results):
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(os.path.join(self.args.output_dir, fname), "w") as f:
            [f.write(f"{k} = {v}\n") for k, v in sorted(results.items())]


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
        print(
            "CUDA not available, using CPU"
            if not torch.cuda.is_available()
            else f"GPU 0: {torch.cuda.get_device_properties(0).name}"
        )
    init_wandb(args)
    hf_train, hf_dev, hf_test, utt_f, int_f, slot_f, is_instr = load_hf_dataset(
        dataset_name=args.hf_dataset,
        cache_dir=args.cache_dir or None,
        dev_split_name=args.dev_split,
        test_split_name=args.test_split,
        train_split_name=args.train_split,
        dev_fraction=args.dev_fraction,
        test_fraction=args.test_fraction,
    )
    intent_label_set, slot_label_set = extract_label_sets(hf_train, int_f, slot_f, is_instr)
    tokenizer = setup_tokenizer(args.model_name_or_path)
    intent_pos_weight = None
    if not getattr(args, "disable_intent_class_balance", False):
        intent_pos_weight = compute_intent_pos_weight(
            hf_train,
            int_f,
            is_instr,
            intent_label_set,
            max_weight=getattr(args, "intent_pos_weight_max", 50.0),
        )
        if _wandb_active:
            wb.log(
                {
                    "intent_pos_weight/min": intent_pos_weight.min().item(),
                    "intent_pos_weight/max": intent_pos_weight.max().item(),
                    "intent_pos_weight/mean": intent_pos_weight.mean().item(),
                },
                step=0,
            )
    else:
        logger.warning(
            "--disable_intent_class_balance set: intent BCE runs WITHOUT pos_weight. For a compound-intent, class-imbalanced target space this is very likely to reproduce the 'loss falls, subset accuracy stays at chance' pattern."
        )
    freq_index = WordFrequencyIndex(
        smoothing=getattr(args, "freq_smoothing", 0.5), min_freq=getattr(args, "freq_min_count", 1)
    )
    freq_index.build(hf_train, utt_f, is_instr)
    if _wandb_active:
        wb.log(freq_index.summary_stats(), step=0)
    make_ds = lambda split_data: HFSLUDataset(
        args=args,
        hf_split=split_data,
        utterance_field=utt_f,
        intent_field=int_f,
        slot_field=slot_f,
        intent_label_set=intent_label_set,
        slot_label_set=slot_label_set,
        tokenizer=tokenizer,
        is_instruction=is_instr,
        freq_index=freq_index,
    )
    trainer = EarlyExitTrainer(
        args=args,
        tokenizer=tokenizer,
        intent_pos_weight=intent_pos_weight,
        train_ds=make_ds(hf_train) if args.do_train else None,
        dev_ds=make_ds(hf_dev),
        test_ds=make_ds(hf_test),
        intent_label_set=intent_label_set,
        slot_label_set=slot_label_set,
    )
    if args.do_train:
        trainer.train()
    if args.do_eval:
        trainer.load_model()
        if getattr(args, "calibrate_exit", False):
            trainer.calibrate_exit_hparams()
        trainer.evaluate("test")


def build_argparser():
    import argparse

    p = argparse.ArgumentParser(
        description="Partially-frozen-backbone joint intent detection + BIO slot filling, with frequency-adaptive PABEE early exit (no BiSLU, no self-distillation, no SCL).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--hf_dataset", required=True)
    p.add_argument("--cache_dir", default="")
    p.add_argument("--train_split", default="train")
    p.add_argument("--dev_split", default="validation")
    p.add_argument("--test_split", default="test")
    p.add_argument("--dev_fraction", default=0.1, type=float)
    p.add_argument("--test_fraction", default=0.1, type=float)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_eval", action="store_true")
    p.add_argument("--max_seq_length", default=100, type=int)
    p.add_argument("--train_batch_size", default=16, type=int)
    p.add_argument("--eval_batch_size", default=8, type=int)
    p.add_argument(
        "--learning_rate",
        default=0.0002,
        type=float,
        help="Only per-layer heads are trained; a higher LR than full fine-tuning is typical for frozen-backbone probing.",
    )
    p.add_argument("--num_train_epochs", default=15, type=int)
    p.add_argument("--warmup_proportion", default=0.1, type=float)
    p.add_argument("--gradient_accumulation_steps", default=2, type=int)
    p.add_argument("--weight_decay", default=0.01, type=float)
    p.add_argument("--adam_epsilon", default=1e-08, type=float)
    p.add_argument("--max_grad_norm", default=1.0, type=float)
    p.add_argument("--logging_steps", default=200, type=int)
    p.add_argument("--early_stopping", default=5, type=int)
    p.add_argument(
        "--tuning_metric",
        default="mean_f1",
        help="Metric used for early stopping, checkpointing, and (unless overridden) HPO. Use 'balanced_score' to jointly optimize intent_acc, mean_f1, semantic_acc, and exit-layer compute savings via --balanced_weight_* (a small mean_f1 drop is accepted in exchange for a larger layer_savings_pct gain, per the configured weights).",
    )
    p.add_argument(
        "--balanced_weight_intent_acc",
        default=0.3,
        type=float,
        help="Weight on intent_acc in the 'balanced_score' tuning metric.",
    )
    p.add_argument(
        "--balanced_weight_mean_f1",
        default=0.25,
        type=float,
        help="Weight on mean_f1 in the 'balanced_score' tuning metric.",
    )
    p.add_argument(
        "--balanced_weight_semantic_acc",
        default=0.2,
        type=float,
        help="Weight on semantic_acc in the 'balanced_score' tuning metric.",
    )
    p.add_argument(
        "--balanced_weight_efficiency",
        default=0.25,
        type=float,
        help="Weight on layer_savings_pct (1 - mean_exit_layer/(num_layers-1)) in the "
        "'balanced_score' tuning metric; raising this trades accuracy/F1 for "
        "earlier average exit.",
    )
    p.add_argument("--loss_coef_intent", default=0.5, type=float)
    p.add_argument("--loss_coef_slot", default=0.5, type=float)
    p.add_argument("--dropout_rate", default=0.1, type=float)
    p.add_argument(
        "--intent_head_hidden",
        default=128,
        type=int,
        help="Hidden width of a single non-linear layer in the per-layer intent probe (0 = pure linear probe, matching strict linear-probing convention). Default 128: the multi-label/compound-intent decision boundary is harder than per-token BIO tagging, see ExitHead docstring.",
    )
    p.add_argument(
        "--slot_head_hidden",
        default=0,
        type=int,
        help="Hidden width for the per-layer slot probe. Default 0 (pure linear): the slot head was already performing well (F1~0.94) as a linear probe, so it is left alone here.",
    )
    p.add_argument(
        "--disable_intent_class_balance",
        action="store_true",
        help="Turn OFF the pos_weight class-imbalance correction on intent BCE (for ablation only -- expect subset accuracy to collapse again).",
    )
    p.add_argument(
        "--intent_pos_weight_max",
        default=50.0,
        type=float,
        help="Clip ceiling for per-class pos_weight = n_negative/n_positive.",
    )
    p.add_argument(
        "--intent_threshold_init",
        default=0.5,
        type=float,
        help="Initial sigmoid threshold before the first dev-set threshold search.",
    )
    p.add_argument(
        "--min_exit_layer",
        default=None,
        type=int,
        help="Hard floor: clamped up to ceil(num_layers/2) if set lower.",
    )
    p.add_argument("--ee_patience", default=3, type=int)
    p.add_argument(
        "--tau_intent",
        default=0.05,
        type=float,
        help="UNUSED by the current exit criterion (kept only for CLI/checkpoint back-compat). Intent stability is now decided by discretized-label agreement + --intent_exit_margin, not by raw probability drift -- see forward_with_early_exit / _discretize_intent docstrings for why the magnitude-delta version could lock onto a stable-but-wrong exit.",
    )
    p.add_argument(
        "--tau_slot",
        default=0.1,
        type=float,
        help="Max fraction of word positions allowed to change predicted BIO tag between consecutive layers to count as slot-stable.",
    )
    p.add_argument(
        "--intent_exit_margin",
        default=0.15,
        type=float,
        help="Confidence gate for the intent exit criterion: a layer's sigmoid output must move at least this far from --intent_threshold_init on at least one class before consecutive-layer label agreement is trusted as 'stable'. Prevents a flat, near-threshold, undertrained head from satisfying patience by agreeing with itself.",
    )
    p.add_argument(
        "--calibrate_exit",
        action="store_true",
        help="After loading the best checkpoint and before the final test evaluation, grid-search (ee_patience, min_exit_layer) on DEV ONLY and freeze the best combination. See calibrate_exit_hparams docstring. No effect unless --do_eval is also set.",
    )
    p.add_argument(
        "--ee_patience_decay",
        default=0.5,
        type=float,
        help="NOVELTY: shrinks the layers-of-agreement required to exit as depth increases past the per-sample minimum (0 = original flat --ee_patience everywhere, i.e. exact V2 behaviour). Targets the low layer_savings_pct a flat patience produces once you're well past min_exit_layer; the confidence-margin + joint intent/slot agreement gate is unchanged, so this only removes conservatism, it doesn't remove the safety check.",
    )
    p.add_argument(
        "--ee_patience_min",
        default=1,
        type=int,
        help="Floor on the depth-adaptive required patience above -- never requires fewer than this many consecutive stable layers to exit.",
    )
    p.add_argument(
        "--disable_exit_logit_smoothing",
        action="store_true",
        help="NOVELTY (on by default): the exit criterion already computes the layer just before the one that triggers exit; averaging its logits with the exiting layer's is a free 2-layer ensemble (zero extra FLOPs) that reduces single-head prediction noise. Pass this flag to use only the exiting layer's own logits (matches V2 behaviour exactly).",
    )
    p.add_argument(
        "--cache_frozen_features",
        action="store_true",
        help="NOVELTY: precompute every layer's pooled (cls, word) features for the frozen backbone ONCE before training, then train all epochs directly off that cache instead of rerunning the (frozen, deterministic) backbone forward pass every step of every epoch. Bit-identical features -> identical gradients -> same results, purely faster. Off by default because memory scales with dataset size x depth x hidden_size; see --cache_max_gb. AUTOMATICALLY DISABLED whenever --unfrozen_ratio makes any backbone layer trainable, since the cache's bit-identical-features assumption only holds for a fully frozen backbone.",
    )
    p.add_argument(
        "--cache_max_gb",
        default=6.0,
        type=float,
        help="Memory budget (GB, fp16) for --cache_frozen_features. If the train split would exceed this, caching is skipped with a warning and training falls back to the normal (recomputed-every-epoch) path -- never crashes with an OOM.",
    )
    p.add_argument(
        "--intent_loss_fn",
        default="asl",
        choices=["asl", "bce"],
        help="NOVELTY: intent classification loss. 'asl' = Asymmetric Loss (Ben-Baruch et al. 2020), a multi-label-imbalance-aware replacement for BCE+pos_weight -- see asymmetric_loss() docstring. 'bce' reproduces the exact original BCE+pos_weight behaviour.",
    )
    p.add_argument(
        "--asl_gamma_neg",
        default=4.0,
        type=float,
        help="ASL negative-class focusing exponent (higher = easy negatives matter less). No effect if --intent_loss_fn bce.",
    )
    p.add_argument(
        "--asl_gamma_pos",
        default=0.0,
        type=float,
        help="ASL positive-class focusing exponent (0 = no down-weighting of hard positives, standard ASL default). No effect if --intent_loss_fn bce.",
    )
    p.add_argument(
        "--asl_clip",
        default=0.05,
        type=float,
        help="ASL negative-probability shift; hard-discards very-easy negatives. No effect if --intent_loss_fn bce.",
    )
    p.add_argument(
        "--train_probe_size",
        default=1000,
        type=int,
        help="Size of the fixed train subsample re-evaluated each epoch with the exact same exit-based forward pass and metric function as dev, so train and dev numbers are directly comparable (see train() loop).",
    )
    p.add_argument(
        "--train_probe_every",
        default=1,
        type=int,
        help="Run the train-probe diagnostic every N epochs (1 = every epoch).",
    )
    p.add_argument(
        "--require_joint_stability",
        type=lambda s: s.lower() != "false",
        default=True,
        help="If False, exit decisions require only intent stability (single-task patience exit, e.g. PABEE-style baselines); if True, require joint intent+slot stability.",
    )
    p.add_argument(
        "--use_freq_exit",
        action="store_true",
        help="Frequency-adaptive per-sample min exit + depth-weighted loss.",
    )
    p.add_argument("--freq_smoothing", default=0.5, type=float)
    p.add_argument("--freq_min_count", default=1, type=int)
    p.add_argument(
        "--unfrozen_ratio",
        default=0.5,
        type=float,
        help="NOVELTY: fraction of INITIAL backbone transformer layers kept TRAINABLE (the remaining, deeper layers stay frozen). Clamped up to >= 0.5 by design: at least the first half of the backbone must adapt during training so the early-exit heads -- which is where most samples actually answer from, since min_exit_layer also sits >= 50%% depth -- read confident, task-adapted representations instead of purely generic pretrained ones. 1.0 = fully unfrozen backbone. See JointModelWithEarlyExit._apply_partial_freeze.",
    )
    p.add_argument(
        "--unfreeze_position",
        default="front",
        choices=[
            "front",
            "none",
            "all",
            "early",
            "middle",
            "late",
            "early+middle",
            "middle+late",
            "early+late",
        ],
        help="NOVELTY (extension for the unfreezing study): which backbone layers --unfrozen_ratio's layer COUNT is applied to. 'front' (default) is the ORIGINAL V4 behaviour -- first N layers. The quarter-based positions support the Layer Position Ablation (early/middle/late/pairwise unions of the backbone's 4 depth quartiles). No effect on --mode train unless explicitly set.",
    )
    p.add_argument(
        "--min_unfrozen_ratio_floor",
        default=0.5,
        type=float,
        help="Floor applied to --unfrozen_ratio before it is converted to a layer count (see JointModelWithEarlyExit.__init__). Kept at the ORIGINAL V4 default of 0.5 for --mode train (backward compatible). The HPO / unfreeze-sweep / ablation experiment runners below explicitly override this to 0.0 so the full [0.0, 1.0] range required by the research spec can actually be measured.",
    )
    p.add_argument(
        "--backbone_learning_rate",
        default=None,
        type=float,
        help="LR for the trainable (unfrozen) backbone layers, kept separate from --learning_rate (used for the per-layer heads). Default: 0.1x --learning_rate. Kept lower because these are pretrained weights, not randomly-initialized heads, and are far more prone to catastrophic forgetting / instability at head-probing LRs.",
    )
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="frozen-pabee-intent-slot")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_watch_freq", default=100, type=int)
    return p


import copy
import gc
import glob
import itertools
import random
import sqlite3
from collections import OrderedDict

try:
    import pandas as pd
except ImportError:
    pd = None
    logger.warning(
        "pandas not installed -> CSV outputs for hpo/ablation/sensitivity modes will be disabled. Run: pip install pandas --break-system-packages"
    )
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clone_args(args, **overrides):
    new_args = copy.deepcopy(args)
    for k, v in overrides.items():
        setattr(new_args, k, v)
    return new_args


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_datasets_bundle(args) -> Dict:
    hf_train, hf_dev, hf_test, utt_f, int_f, slot_f, is_instr = load_hf_dataset(
        dataset_name=args.hf_dataset,
        cache_dir=args.cache_dir or None,
        dev_split_name=args.dev_split,
        test_split_name=args.test_split,
        train_split_name=args.train_split,
        dev_fraction=args.dev_fraction,
        test_fraction=args.test_fraction,
    )
    intent_label_set, slot_label_set = extract_label_sets(hf_train, int_f, slot_f, is_instr)
    tokenizer = setup_tokenizer(args.model_name_or_path)
    intent_pos_weight = None
    if not getattr(args, "disable_intent_class_balance", False):
        intent_pos_weight = compute_intent_pos_weight(
            hf_train,
            int_f,
            is_instr,
            intent_label_set,
            max_weight=getattr(args, "intent_pos_weight_max", 50.0),
        )
    freq_index = WordFrequencyIndex(
        smoothing=getattr(args, "freq_smoothing", 0.5), min_freq=getattr(args, "freq_min_count", 1)
    )
    freq_index.build(hf_train, utt_f, is_instr)
    hpo_frac = float(getattr(args, "hpo_subset_fraction", 1.0))
    if getattr(args, "mode", "train") == "hpo" and 0.0 < hpo_frac < 1.0:
        n = max(10, int(len(hf_train) * hpo_frac))
        hf_train_for_ds = hf_train.shuffle(seed=getattr(args, "hpo_seed", 42)).select(range(n))
        logger.info(
            "HPO subset: using %d / %d train examples (%.0f%%).", n, len(hf_train), hpo_frac * 100
        )
    else:
        hf_train_for_ds = hf_train
    make_ds = lambda split_data: HFSLUDataset(
        args=args,
        hf_split=split_data,
        utterance_field=utt_f,
        intent_field=int_f,
        slot_field=slot_f,
        intent_label_set=intent_label_set,
        slot_label_set=slot_label_set,
        tokenizer=tokenizer,
        is_instruction=is_instr,
        freq_index=freq_index,
    )
    return {
        "train_ds": make_ds(hf_train_for_ds),
        "full_train_ds": make_ds(hf_train),
        "dev_ds": make_ds(hf_dev),
        "test_ds": make_ds(hf_test),
        "tokenizer": tokenizer,
        "intent_label_set": intent_label_set,
        "slot_label_set": slot_label_set,
        "intent_pos_weight": intent_pos_weight,
        "freq_index": freq_index,
    }


def _new_trainer(args, bundle: Dict, use_full_train: bool = False) -> "EarlyExitTrainer":
    return EarlyExitTrainer(
        args=args,
        tokenizer=bundle["tokenizer"],
        train_ds=bundle["full_train_ds"] if use_full_train else bundle["train_ds"],
        dev_ds=bundle["dev_ds"],
        test_ds=bundle["test_ds"],
        intent_label_set=bundle["intent_label_set"],
        slot_label_set=bundle["slot_label_set"],
        intent_pos_weight=bundle["intent_pos_weight"],
    )


def _delete_checkpoint(output_dir: str) -> None:
    ckpt = os.path.join(output_dir, "checkpoint.pth")
    if os.path.exists(ckpt):
        try:
            os.remove(ckpt)
            logger.info("Deleted checkpoint (metrics already extracted): %s", ckpt)
        except OSError as e:
            logger.warning("Could not delete checkpoint %s: %s", ckpt, e)


def estimate_transformer_flops(
    hidden_size: int, num_layers: int, seq_len: int, intermediate_mult: float = 4.0
) -> float:
    d = float(hidden_size)
    per_token_per_layer = 8.0 * d * d + 4.0 * intermediate_mult * d * d
    fwd = per_token_per_layer * num_layers * seq_len
    return fwd


class CostTracker:

    def __init__(self, device: str):
        self.device = device

    def __enter__(self):
        _cleanup_cuda()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.wall_time_sec = time.perf_counter() - self.t0
        if torch.cuda.is_available():
            self.peak_mem_MB = torch.cuda.max_memory_allocated() / 1024**2
        else:
            self.peak_mem_MB = float("nan")
        return False


def run_single_experiment(
    args,
    bundle: Dict,
    seed: int,
    use_full_train: bool = False,
    eval_mode: str = "test",
    delete_ckpt: bool = True,
) -> Dict:
    set_seed(seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = None
    result: Dict = {"failed": False, "seed": seed}
    try:
        with CostTracker(device) as ct:
            trainer = _new_trainer(args, bundle, use_full_train=use_full_train)
            trainer.train()
        trainer.load_model()
        eval_res = trainer.evaluate(eval_mode, log_wandb=False, quiet=True)
        result.update(eval_res)
        result["train_wall_time_sec"] = ct.wall_time_sec
        result["peak_gpu_mem_MB"] = ct.peak_mem_MB
        result["num_unfrozen_layers"] = getattr(trainer.model, "num_unfrozen_layers", None)
        result["num_layers"] = trainer.model.num_layers
        result["est_train_flops_per_sample"] = estimate_transformer_flops(
            hidden_size=(
                trainer.model.wordrep.base_model.config.hidden_size
                if hasattr(trainer.model.wordrep.base_model, "config")
                else 0
            ),
            num_layers=trainer.model.num_layers,
            seq_len=args.max_seq_length,
        )
        if "mean_exit_layer" in eval_res:
            result["est_inference_flops_per_sample"] = estimate_transformer_flops(
                hidden_size=(
                    trainer.model.wordrep.base_model.config.hidden_size
                    if hasattr(trainer.model.wordrep.base_model, "config")
                    else 0
                ),
                num_layers=max(1, int(round(eval_res["mean_exit_layer"])) + 1),
                seq_len=args.max_seq_length,
            )
    except RuntimeError as e:
        logger.error("Experiment failed [output_dir=%s seed=%d]: %s", args.output_dir, seed, e)
        result["failed"] = True
        result["error"] = str(e)
    finally:
        if delete_ckpt:
            _delete_checkpoint(args.output_dir)
        if trainer is not None:
            del trainer
        _cleanup_cuda()
    return result


UNFREEZE_RATIO_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
UNFREEZE_POSITION_GRID = [
    "early",
    "middle",
    "late",
    "early+middle",
    "middle+late",
    "early+late",
    "all",
    "none",
]


def run_unfreeze_ratio_sweep(
    base_args,
    bundle: Dict,
    seeds: Tuple[int, ...] = (42, 43, 44),
    ratios: Optional[List[float]] = None,
) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas required for run_unfreeze_ratio_sweep.")
    ratios = ratios or UNFREEZE_RATIO_GRID
    records: List[Dict] = []
    for ratio in ratios:
        for seed in seeds:
            trial_args = _clone_args(
                base_args,
                unfrozen_ratio=ratio,
                unfreeze_position="front",
                min_unfrozen_ratio_floor=0.0,
                use_wandb=False,
            )
            trial_args.output_dir = os.path.join(
                base_args.output_dir, "unfreeze", "ratio", f"r{ratio:.1f}_seed{seed}"
            )
            logger.info("=== Unfreeze ratio=%.1f seed=%d ===", ratio, seed)
            res = run_single_experiment(trial_args, bundle, seed, eval_mode="test")
            res.update({"unfrozen_ratio": ratio})
            records.append(res)
    df = pd.DataFrame(records)
    os.makedirs(base_args.output_dir, exist_ok=True)
    out_dir = os.path.join(base_args.output_dir, "unfreeze")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "layer_ratio_results.csv"), index=False)
    return df


def run_unfreeze_position_sweep(
    base_args,
    bundle: Dict,
    seeds: Tuple[int, ...] = (42, 43, 44),
    positions: Optional[List[str]] = None,
) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas required for run_unfreeze_position_sweep.")
    positions = positions or UNFREEZE_POSITION_GRID
    records: List[Dict] = []
    for pos in positions:
        for seed in seeds:
            trial_args = _clone_args(
                base_args,
                unfrozen_ratio=1.0,
                unfreeze_position=pos,
                min_unfrozen_ratio_floor=0.0,
                use_wandb=False,
            )
            trial_args.output_dir = os.path.join(
                base_args.output_dir, "unfreeze", "position", f"{pos.replace('+', '_')}_seed{seed}"
            )
            logger.info("=== Unfreeze position=%s seed=%d ===", pos, seed)
            res = run_single_experiment(trial_args, bundle, seed, eval_mode="test")
            res.update({"unfreeze_position": pos})
            records.append(res)
    df = pd.DataFrame(records)
    out_dir = os.path.join(base_args.output_dir, "unfreeze")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "layer_position_results.csv"), index=False)
    return df


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic = torch.norm(Y.T @ X, p="fro") ** 2
    denom = torch.norm(X.T @ X, p="fro") * torch.norm(Y.T @ Y, p="fro")
    if denom.item() == 0.0:
        return float("nan")
    return (hsic / denom).item()


@torch.no_grad()
def compute_layerwise_cka(
    model_a: "JointModelWithEarlyExit",
    model_b: "JointModelWithEarlyExit",
    ds,
    tokenizer,
    device: str,
    n_samples: int = 128,
) -> List[float]:
    n = min(n_samples, len(ds))
    idx = list(range(n))
    batch = [ds[i] for i in idx]
    iids, amask, wlen, wattn, il, sl, freq = collate_fn(batch, tokenizer.pad_token_id)
    iids, amask = (iids.to(device), amask.to(device))
    model_a.eval()
    model_b.eval()
    out_a = model_a.wordrep.base_model(iids, attention_mask=amask, output_hidden_states=True)
    out_b = model_b.wordrep.base_model(iids, attention_mask=amask, output_hidden_states=True)
    L = min(len(out_a.hidden_states), len(out_b.hidden_states))
    cka_per_layer = []
    for l in range(1, L):
        Ha = out_a.hidden_states[l].mean(dim=1).float().cpu()
        Hb = out_b.hidden_states[l].mean(dim=1).float().cpu()
        cka_per_layer.append(_linear_cka(Ha, Hb))
    return cka_per_layer


def load_trained_model(
    args, bundle: Dict, checkpoint_dir: str, device: str
) -> "JointModelWithEarlyExit":
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    model = JointModelWithEarlyExit(
        args, len(bundle["intent_label_set"]), len(bundle["slot_label_set"])
    ).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    return model


def compute_gradient_norms_per_layer(
    args, bundle: Dict, checkpoint_dir: str, n_batches: int = 5, seed: int = 42
) -> List[float]:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trial_args = _clone_args(args, use_wandb=False)
    trainer = _new_trainer(trial_args, bundle)
    model = load_trained_model(trial_args, bundle, checkpoint_dir, device)
    trainer.model = model
    model.train()
    dl = trainer._dl(trainer.train_ds, True)
    bm = model.wordrep.base_model
    n_layers = len(bm.layers) if hasattr(bm, "layers") else 0
    grad_norm_sums = [0.0] * n_layers
    n_steps = 0
    for step, batch in enumerate(dl):
        if step >= n_batches:
            break
        inputs = {
            "input_ids": batch[0].to(device),
            "attention_mask": batch[1].to(device),
            "words_lengths": batch[2].to(device),
            "word_attention_mask": batch[3].to(device),
        }
        slot_labels, intent_labels, word_attn = (
            batch[5].to(device),
            batch[4].to(device),
            batch[3].to(device),
        )
        loss, *_ = trainer.compute_loss(
            model, inputs, slot_labels, intent_labels, word_attn, freq_scores=batch[6].to(device)
        )
        model.zero_grad(set_to_none=True)
        loss.backward()
        if hasattr(bm, "layers"):
            for i, layer in enumerate(bm.layers):
                norms = [p.grad.norm().item() for p in layer.parameters() if p.grad is not None]
                if norms:
                    grad_norm_sums[i] += float(np.mean(norms))
        n_steps += 1
    model.zero_grad(set_to_none=True)
    del trainer, model
    _cleanup_cuda()
    if n_steps == 0:
        return grad_norm_sums
    return [s / n_steps for s in grad_norm_sums]


def run_layer_contribution_analysis(
    base_args,
    bundle: Dict,
    best_ratio: float,
    trained_checkpoint_dir: str,
    best_position: str = "front",
) -> Dict:
    if pd is None:
        raise RuntimeError("pandas required for run_layer_contribution_analysis.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_args = _clone_args(
        base_args,
        unfrozen_ratio=0.0,
        unfreeze_position="none",
        min_unfrozen_ratio_floor=0.0,
        use_wandb=False,
    )
    trained_args = _clone_args(
        base_args,
        unfrozen_ratio=best_ratio,
        unfreeze_position=best_position,
        min_unfrozen_ratio_floor=0.0,
        use_wandb=False,
    )
    set_seed(42)
    model_pretrained = JointModelWithEarlyExit(
        pretrained_args, len(bundle["intent_label_set"]), len(bundle["slot_label_set"])
    ).to(device)
    model_trained = load_trained_model(trained_args, bundle, trained_checkpoint_dir, device)
    cka_per_layer = compute_layerwise_cka(
        model_pretrained, model_trained, bundle["dev_ds"], bundle["tokenizer"], device
    )
    del model_pretrained, model_trained
    _cleanup_cuda()
    grad_norms = compute_gradient_norms_per_layer(trained_args, bundle, trained_checkpoint_dir)
    df = pd.DataFrame(
        {
            "layer": list(range(len(cka_per_layer))),
            "cka_pretrained_vs_trained": cka_per_layer,
            "representation_drift": [1.0 - c if c == c else float("nan") for c in cka_per_layer],
            "grad_norm_mean": (
                grad_norms + [float("nan")] * (len(cka_per_layer) - len(grad_norms))
                if grad_norms
                else [float("nan")] * len(cka_per_layer)
            ),
        }
    )
    df["trainable"] = df["layer"] >= len(cka_per_layer) - int(
        round(best_ratio * len(cka_per_layer))
    )
    out_dir = os.path.join(base_args.output_dir, "unfreeze")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "layer_contribution.csv"), index=False)
    summary = {
        "best_ratio": best_ratio,
        "best_position": best_position,
        "lowest_cka_layer": int(df["cka_pretrained_vs_trained"].idxmin()) if not df.empty else None,
        "highest_grad_layer": (
            int(df["grad_norm_mean"].idxmax()) if not df["grad_norm_mean"].isna().all() else None
        ),
    }
    with open(os.path.join(out_dir, "unfreezing_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return {"layer_contribution_df": df, "summary": summary}


try:
    import optuna
    from optuna.samplers import TPESampler

    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    logger.warning(
        "optuna not installed -> --mode hpo disabled. Run: pip install optuna --break-system-packages"
    )


def _suggest_hpo_params(trial, base_args) -> Dict:
    num_layers_guess = getattr(base_args, "_hpo_num_layers_hint", 16)
    hp: Dict = {
        "unfrozen_ratio": trial.suggest_float("unfrozen_ratio", 0.1, 1.0),
        "intent_head_hidden": trial.suggest_categorical("intent_head_hidden", [0, 64, 128, 256]),
        "slot_head_hidden": trial.suggest_categorical("slot_head_hidden", [0, 64, 128, 256]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.05, 0.4),
        "backbone_learning_rate": trial.suggest_float(
            "learning_rate_backbone", 1e-05, 0.001, log=True
        ),
        "learning_rate": trial.suggest_float("learning_rate_head", 0.0001, 0.01, log=True),
        "warmup_proportion": trial.suggest_float("warmup_proportion", 0.03, 0.2),
        "weight_decay": trial.suggest_float("weight_decay", 1e-05, 0.1, log=True),
        "train_batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "loss_coef_intent": trial.suggest_float("loss_coef_intent", 0.3, 0.7),
        "loss_coef_slot": trial.suggest_float("loss_coef_slot", 0.3, 0.7),
        "intent_loss_fn": trial.suggest_categorical("intent_loss_fn", ["asl", "bce"]),
        "min_exit_layer": trial.suggest_int(
            "min_exit_layer", max(1, math.ceil(num_layers_guess / 4)), max(2, num_layers_guess - 1)
        ),
        "ee_patience": trial.suggest_int("ee_patience", 1, 6),
        "ee_patience_decay": trial.suggest_float("ee_patience_decay", 0.0, 1.0),
        "tau_slot": trial.suggest_float("tau_slot", 0.02, 0.3, log=True),
        "intent_exit_margin": trial.suggest_float("intent_exit_margin", 0.05, 0.3),
        "use_freq_exit": trial.suggest_categorical("use_freq_exit", [True, False]),
    }
    if hp["intent_loss_fn"] == "asl":
        hp["asl_gamma_neg"] = trial.suggest_float("asl_gamma_neg", 1.0, 6.0)
        hp["asl_clip"] = trial.suggest_float("asl_clip", 0.01, 0.1)
    if hp["use_freq_exit"]:
        hp["freq_smoothing"] = trial.suggest_float("freq_smoothing", 0.1, 1.0)
        hp["freq_min_count"] = trial.suggest_int("freq_min_count", 1, 5)
    hp["gradient_accumulation_steps"] = max(1, 32 // hp["train_batch_size"])
    hp["min_unfrozen_ratio_floor"] = 0.0
    return hp


def _hpo_objective(trial, base_args, bundle: Dict) -> float:
    hp = _suggest_hpo_params(trial, base_args)
    trial_args = _clone_args(base_args, **hp)
    trial_args.num_train_epochs = getattr(base_args, "hpo_epochs", 4)
    trial_args.early_stopping = getattr(base_args, "hpo_early_stopping", 2)
    trial_args.use_wandb = False
    trial_args.output_dir = os.path.join(
        base_args.output_dir, "hpo_trials", f"trial_{trial.number}"
    )
    result = run_single_experiment(
        trial_args,
        bundle,
        seed=getattr(base_args, "hpo_seed", 42),
        eval_mode="dev",
        delete_ckpt=True,
    )
    if result.get("failed"):
        raise optuna.TrialPruned(f"Trial {trial.number} raised RuntimeError: {result.get('error')}")
    score = result.get(trial_args.tuning_metric, float("nan"))
    if not math.isfinite(score):
        raise optuna.TrialPruned(
            f"Non-finite {trial_args.tuning_metric}={score!r} for trial {trial.number}."
        )
    return score


def run_hpo(base_args, bundle: Dict) -> Optional[Dict]:
    if not _OPTUNA_AVAILABLE:
        logger.error("Cannot run HPO: optuna is not installed.")
        return None
    os.makedirs(base_args.output_dir, exist_ok=True)
    cfg = AutoConfig.from_pretrained(base_args.model_name_or_path)
    base_args._hpo_num_layers_hint = cfg.num_hidden_layers
    storage_path = os.path.join(base_args.output_dir, "optuna_study.db")
    if os.path.exists(storage_path):
        try:
            conn = sqlite3.connect(storage_path)
            status = conn.execute("PRAGMA integrity_check;").fetchone()
            conn.close()
            if status[0] != "ok":
                logger.warning("Optuna SQLite DB failed integrity_check; starting fresh.")
                os.remove(storage_path)
        except sqlite3.Error as e:
            logger.warning("SQLite integrity check raised %s; starting fresh.", e)
            if os.path.exists(storage_path):
                os.remove(storage_path)
    study = optuna.create_study(
        study_name=getattr(base_args, "study_name", "intent_v5_hpo"),
        storage=f"sqlite:///{storage_path}",
        direction="maximize",
        load_if_exists=True,
        sampler=TPESampler(seed=getattr(base_args, "hpo_seed", 42)),
    )
    n_trials = getattr(base_args, "n_trials", 30)
    study.optimize(
        lambda trial: _hpo_objective(trial, base_args, bundle),
        n_trials=n_trials,
        catch=(RuntimeError,),
        gc_after_trial=True,
    )
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.error("HPO finished with zero completed trials out of %d requested.", n_trials)
        return None
    best_params, best_value = (study.best_params, study.best_value)
    logger.info(
        "HPO complete: best %s = %.4f | params = %s",
        base_args.tuning_metric,
        best_value,
        best_params,
    )
    translated = dict(best_params)
    if "learning_rate_backbone" in translated:
        translated["backbone_learning_rate"] = translated.pop("learning_rate_backbone")
    if "learning_rate_head" in translated:
        translated["learning_rate"] = translated.pop("learning_rate_head")
    if "batch_size" in translated:
        translated["train_batch_size"] = translated.pop("batch_size")
        translated["gradient_accumulation_steps"] = max(1, 32 // translated["train_batch_size"])
    translated["min_unfrozen_ratio_floor"] = 0.0
    out_dir = os.path.join(base_args.output_dir, "hpo")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(
            {
                "best_value": best_value,
                "best_params_raw": best_params,
                "best_params_translated": translated,
                "tuning_metric": base_args.tuning_metric,
                "n_trials_completed": len(completed),
                "n_trials_total": len(study.trials),
            },
            f,
            indent=2,
        )
    if pd is not None:
        trials_df = study.trials_dataframe()
        trials_df.to_csv(os.path.join(out_dir, "hpo_trials.csv"), index=False)
    return translated


def build_ablation_experiments(
    best_params: Optional[Dict] = None, num_layers: Optional[int] = None
) -> "OrderedDict[str, Dict]":
    bp = best_params or {}
    full = {
        "use_freq_exit": bp.get("use_freq_exit", True),
        "intent_loss_fn": bp.get("intent_loss_fn", "asl"),
        "ee_patience_decay": bp.get("ee_patience_decay", 0.5),
        "exit_logit_smoothing": True,
        "require_joint_stability": True,
        "unfrozen_ratio": bp.get("unfrozen_ratio", 0.5),
        "min_unfrozen_ratio_floor": 0.0,
    }
    exps = OrderedDict()
    exps["E0_full_model"] = dict(full)
    exps["E1_no_freq_adaptive_exit"] = {**full, "use_freq_exit": False}
    exps["E2_bce_instead_of_asl"] = {**full, "intent_loss_fn": "bce"}
    exps["E3_no_depth_adaptive_patience"] = {**full, "ee_patience_decay": 0.0}
    exps["E4_no_exit_logit_smoothing"] = {**full, "exit_logit_smoothing": False}
    exps["E5_fully_frozen_backbone"] = {**full, "unfrozen_ratio": 0.0}
    exps["E6_single_task_exit"] = {**full, "require_joint_stability": False}
    return exps


def build_baseline_experiments(
    best_params: Optional[Dict] = None, num_layers: int = 16
) -> "OrderedDict[str, Dict]":
    bp = best_params or {}
    common = {
        "intent_loss_fn": bp.get("intent_loss_fn", "asl"),
        "unfrozen_ratio": bp.get("unfrozen_ratio", 0.5),
        "min_unfrozen_ratio_floor": 0.0,
        "ee_patience_decay": bp.get("ee_patience_decay", 0.5),
        "exit_logit_smoothing": True,
    }
    exps = OrderedDict()
    exps["B0_full_depth"] = {
        **common,
        "use_freq_exit": False,
        "require_joint_stability": False,
        "min_exit_layer": num_layers - 1,
        "ee_patience": 10**6,
        "ee_patience_decay": 0.0,
    }
    exps["B1_confidence_exit"] = {
        **common,
        "use_freq_exit": False,
        "require_joint_stability": False,
        "min_exit_layer": max(1, num_layers // 4),
        "ee_patience": 1,
        "ee_patience_decay": 0.0,
    }
    exps["B2_deebert_style"] = {
        **common,
        "use_freq_exit": False,
        "require_joint_stability": False,
        "min_exit_layer": max(1, num_layers // 4),
        "ee_patience": 2,
        "ee_patience_decay": 0.0,
    }
    exps["B3_pabee"] = {
        **common,
        "use_freq_exit": False,
        "require_joint_stability": False,
        "min_exit_layer": max(1, num_layers // 4),
    }
    exps["B4_pabee_joint"] = {
        **common,
        "use_freq_exit": False,
        "require_joint_stability": True,
        "min_exit_layer": max(1, num_layers // 4),
    }
    exps["M1_frequency_unaware"] = {
        **common,
        "use_freq_exit": False,
        "require_joint_stability": True,
    }
    exps["M4_full_method"] = {
        **common,
        "use_freq_exit": bp.get("use_freq_exit", True),
        "require_joint_stability": True,
    }
    return exps


def run_baseline_study(
    base_args,
    bundle: Dict,
    seeds: Tuple[int, ...] = (42, 43, 44),
    best_params: Optional[Dict] = None,
) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas required for run_baseline_study.")
    num_layers = getattr(base_args, "num_backbone_layers", 16)
    experiments = build_baseline_experiments(best_params, num_layers=num_layers)
    records: List[Dict] = []
    for exp_name, flags in experiments.items():
        for seed in seeds:
            exp_args = _clone_args(base_args, **flags)
            exp_args.output_dir = os.path.join(
                base_args.output_dir, "baselines", exp_name, f"seed{seed}"
            )
            exp_args.use_wandb = False
            logger.info("=== Baseline %s | seed=%d | flags=%s ===", exp_name, seed, flags)
            res = run_single_experiment(exp_args, bundle, seed, eval_mode="test")
            res.update({"experiment": exp_name, **flags})
            records.append(res)
    df = pd.DataFrame(records)
    out_dir = os.path.join(base_args.output_dir, "baselines")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "baseline_results.csv"), index=False)
    return df


def run_ablation_study(
    base_args,
    bundle: Dict,
    seeds: Tuple[int, ...] = (42, 43, 44),
    best_params: Optional[Dict] = None,
    experiments: Optional["OrderedDict[str, Dict]"] = None,
) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas required for run_ablation_study.")
    experiments = experiments or build_ablation_experiments(best_params)
    records: List[Dict] = []
    for exp_name, flags in experiments.items():
        for seed in seeds:
            exp_args = _clone_args(base_args, **flags)
            exp_args.output_dir = os.path.join(
                base_args.output_dir, "ablation", exp_name, f"seed{seed}"
            )
            exp_args.use_wandb = False
            logger.info("=== Ablation %s | seed=%d | flags=%s ===", exp_name, seed, flags)
            res = run_single_experiment(exp_args, bundle, seed, eval_mode="test")
            res.update({"experiment": exp_name, **flags})
            records.append(res)
    df = pd.DataFrame(records)
    out_dir = os.path.join(base_args.output_dir, "ablation")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "ablation_results.csv"), index=False)
    return df


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = (np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    n1, n2 = (len(a), len(b))
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
    thresh = np.arange(1, m + 1) / m * alpha
    below = ranked <= thresh
    reject_sorted = np.zeros(m, dtype=bool)
    if below.any():
        max_k = int(np.max(np.nonzero(below)[0]))
        reject_sorted[: max_k + 1] = True
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject.tolist()


def run_statistical_tests(
    df: "pd.DataFrame", group_col: str, baseline: str, metric: str
) -> "pd.DataFrame":
    from scipy import stats

    clean = df[~df.get("failed", False).astype(bool)] if "failed" in df.columns else df
    base_vals = clean.loc[clean[group_col] == baseline, metric].dropna().values
    if len(base_vals) < 2:
        logger.warning(
            "Baseline %s=%r has <2 valid seeds for metric %s; significance tests will be marked non-computable.",
            group_col,
            baseline,
            metric,
        )
    rows: List[Dict] = []
    for name, grp in clean.groupby(group_col):
        if name == baseline:
            continue
        vals = grp[metric].dropna().values
        if len(vals) < 2 or len(base_vals) < 2:
            rows.append(
                {
                    group_col: name,
                    "n": len(vals),
                    "note": "n<2 seeds: significance test not computable",
                }
            )
            continue
        paired = len(vals) == len(base_vals)
        if paired:
            t_stat, t_p = stats.ttest_rel(vals, base_vals)
        else:
            t_stat, t_p = stats.ttest_ind(vals, base_vals, equal_var=False)
        rows.append(
            {
                group_col: name,
                "n": len(vals),
                "paired": paired,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "baseline_mean": float(np.mean(base_vals)),
                "delta": float(np.mean(vals) - np.mean(base_vals)),
                "t_stat": float(t_stat),
                "t_pvalue": float(t_p),
                "cohens_d": _cohens_d(vals, base_vals),
            }
        )
    res_df = pd.DataFrame(rows)
    if not res_df.empty and "t_pvalue" in res_df.columns:
        res_df["t_pvalue_fdr_reject_at_0.05"] = _bh_fdr(res_df["t_pvalue"].fillna(1.0).tolist())
    logger.warning(
        "Statistical tests computed with n=%d seed(s) for the baseline arm -- treat p-values as indicative, not confirmatory, below n~10.",
        len(base_vals),
    )
    return res_df


SENSITIVITY_GRID: "OrderedDict[str, List]" = OrderedDict(
    [
        ("unfrozen_ratio", [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
        ("learning_rate", [0.0001, 0.0003, 0.001, 0.003, 0.01]),
        ("ee_patience", [1, 2, 3, 4, 5, 6]),
        ("tau_slot", [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]),
        ("intent_exit_margin", [0.05, 0.1, 0.15, 0.2, 0.3]),
        ("loss_coef_intent", [0.3, 0.4, 0.5, 0.6, 0.7]),
        ("dropout_rate", [0.05, 0.1, 0.2, 0.3, 0.4]),
        ("weight_decay", [1e-05, 0.0001, 0.001, 0.01, 0.1]),
        ("min_exit_layer", None),
        ("backbone_learning_rate", [1e-05, 3e-05, 0.0001, 0.0003, 0.001]),
    ]
)


def run_sensitivity_analysis(
    base_args, bundle: Dict, center_params: Optional[Dict] = None
) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    if pd is None:
        raise RuntimeError("pandas required for run_sensitivity_analysis.")
    center = dict(center_params or {})
    cfg = AutoConfig.from_pretrained(base_args.model_name_or_path)
    L = cfg.num_hidden_layers
    grid_source = OrderedDict(SENSITIVITY_GRID)
    grid_source["min_exit_layer"] = sorted(
        set((int(round(x)) for x in np.linspace(math.ceil(L / 4), L - 1, 6)))
    )
    records: List[Dict] = []
    for param, grid in grid_source.items():
        for val in grid:
            trial_args = _clone_args(base_args, **center)
            setattr(trial_args, param, val)
            trial_args.min_unfrozen_ratio_floor = 0.0
            trial_args.num_train_epochs = getattr(base_args, "sensitivity_epochs", 3)
            trial_args.early_stopping = getattr(base_args, "sensitivity_early_stopping", 2)
            trial_args.use_wandb = False
            trial_args.output_dir = os.path.join(
                base_args.output_dir, "sensitivity", param, str(val)
            )
            res = run_single_experiment(
                trial_args, bundle, seed=getattr(base_args, "seed", 42), eval_mode="dev"
            )
            score = res.get(base_args.tuning_metric, float("nan"))
            records.append(
                {"param": param, "value": val, "score": score, "failed": res.get("failed", False)}
            )
    df = pd.DataFrame(records)
    out_dir = os.path.join(base_args.output_dir, "sensitivity")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "sensitivity_results.csv"), index=False)
    from scipy import stats as _stats

    corr_rows: List[Dict] = []
    for param, grp in df.groupby("param"):
        clean = grp.dropna(subset=["score"])
        if len(clean) < 3:
            corr_rows.append(
                {
                    "param": param,
                    "spearman_rho": float("nan"),
                    "spearman_p": float("nan"),
                    "n": len(clean),
                }
            )
            continue
        rho, p = _stats.spearmanr(clean["value"].astype(float), clean["score"].astype(float))
        corr_rows.append({"param": param, "spearman_rho": rho, "spearman_p": p, "n": len(clean)})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(out_dir, "sensitivity_correlations.csv"), index=False)
    return (df, corr_df)


def compute_rarity_depth_correlation(
    exit_layers: List[int], freq_scores: List[float]
) -> Tuple["pd.DataFrame", Dict]:
    if pd is None:
        raise RuntimeError("pandas required for compute_rarity_depth_correlation.")
    from scipy import stats as _stats

    exit_layers = np.asarray(exit_layers, dtype=float)
    freq_scores = np.asarray(freq_scores, dtype=float)
    rarity = 1.0 - freq_scores
    mask = np.isfinite(exit_layers) & np.isfinite(rarity)
    exit_layers, rarity = (exit_layers[mask], rarity[mask])
    n = len(rarity)
    if n < 3:
        stats_summary = {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "n": n,
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
        return (pd.DataFrame(), stats_summary)
    pearson_r, pearson_p = _stats.pearsonr(rarity, exit_layers)
    spearman_rho, spearman_p = _stats.spearmanr(rarity, exit_layers)
    z = np.arctanh(np.clip(pearson_r, -0.999999, 0.999999))
    se = 1.0 / math.sqrt(n - 3) if n > 3 else float("nan")
    ci_low, ci_high = (np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se))
    stats_summary = {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "n": n,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "effect_size_r": float(pearson_r),
    }
    quartile_edges = np.quantile(rarity, [0.0, 0.25, 0.5, 0.75, 1.0])
    quartile_idx = np.clip(np.digitize(rarity, quartile_edges[1:-1]), 0, 3)
    rows = []
    for q in range(4):
        qmask = quartile_idx == q
        rows.append(
            {
                "rarity_quartile": f"Q{q + 1}",
                "mean_exit_layer": (
                    float(exit_layers[qmask].mean()) if qmask.any() else float("nan")
                ),
                "n": int(qmask.sum()),
            }
        )
    quartile_df = pd.DataFrame(rows)
    return (quartile_df, stats_summary)


def run_rarity_validation(base_args, bundle: Dict, final_trainer: "EarlyExitTrainer") -> Dict:
    out_dir = os.path.join(base_args.output_dir, "rarity_validation")
    os.makedirs(out_dir, exist_ok=True)
    quartile_df, stats_summary = compute_rarity_depth_correlation(
        final_trainer.last_exit_layers, final_trainer.last_freq_scores
    )
    if not quartile_df.empty:
        quartile_df.to_csv(os.path.join(out_dir, "rarity_quartiles.csv"), index=False)
    with open(os.path.join(out_dir, "rarity_depth_correlation.json"), "w") as f:
        json.dump(stats_summary, f, indent=2)
    return {"quartile_df": quartile_df, "stats": stats_summary}


def run_pareto_operating_points(
    base_args,
    bundle: Dict,
    trained_checkpoint_dir: str,
    min_exit_layers: Optional[List[int]] = None,
    patiences: Optional[List[int]] = None,
) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas required for run_pareto_operating_points.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = AutoConfig.from_pretrained(base_args.model_name_or_path)
    L = cfg.num_hidden_layers
    if min_exit_layers is None:
        min_exit_layers = sorted(
            set((int(round(x)) for x in np.linspace(math.ceil(L / 4), L - 1, 6)))
        )
    if patiences is None:
        patiences = [1, 2, 3, 4, 5]
    records: List[Dict] = []
    for min_exit in min_exit_layers:
        for patience in patiences:
            op_args = _clone_args(
                base_args, min_exit_layer=min_exit, ee_patience=patience, use_wandb=False
            )
            trainer = _new_trainer(op_args, bundle)
            model = load_trained_model(op_args, bundle, trained_checkpoint_dir, device)
            trainer.model = model
            latency = benchmark_latency_throughput(op_args, bundle, model, device, batch_sizes=(1,))
            test_results = trainer.evaluate("test")
            exit_layers = np.asarray(trainer.last_exit_layers, dtype=float)
            record = dict(test_results)
            record.update(
                {
                    "min_exit_layer": min_exit,
                    "ee_patience": patience,
                    "mean_exit_layer": (
                        float(exit_layers.mean()) if exit_layers.size else float("nan")
                    ),
                    "median_exit_layer": (
                        float(np.median(exit_layers)) if exit_layers.size else float("nan")
                    ),
                    "p25_exit_layer": (
                        float(np.percentile(exit_layers, 25)) if exit_layers.size else float("nan")
                    ),
                    "p75_exit_layer": (
                        float(np.percentile(exit_layers, 75)) if exit_layers.size else float("nan")
                    ),
                    "normalized_flops": estimate_transformer_flops(
                        cfg.hidden_size,
                        float(exit_layers.mean()) if exit_layers.size else L,
                        getattr(op_args, "max_seq_length", 128),
                    )
                    / estimate_transformer_flops(
                        cfg.hidden_size, L, getattr(op_args, "max_seq_length", 128)
                    ),
                    "latency_ms_bs1": latency.get("latency_ms_bs1", float("nan")),
                    "throughput_bs1": latency.get("throughput_bs1", float("nan")),
                    "peak_mem_MB_bs1": latency.get("peak_mem_MB_bs1", float("nan")),
                }
            )
            records.append(record)
            del trainer, model
            _cleanup_cuda()
    df = pd.DataFrame(records)
    out_dir = os.path.join(base_args.output_dir, "pareto")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "pareto_operating_points.csv"), index=False)
    return df


def benchmark_latency_throughput(
    args,
    bundle: Dict,
    model: "JointModelWithEarlyExit",
    device: str,
    batch_sizes: Tuple[int, ...] = (1, 8, 32),
    n_warmup: int = 5,
    n_repeats: int = 20,
) -> Dict:
    model.eval()
    ds = bundle["test_ds"]
    pad_id = bundle["tokenizer"].pad_token_id
    results: Dict = {}
    for bs in batch_sizes:
        batch = [ds[i % len(ds)] for i in range(bs)]
        iids, amask, wlen, wattn, _, _, freq = collate_fn(batch, pad_id)
        iids, amask, wlen, wattn = (
            iids.to(device),
            amask.to(device),
            wlen.to(device),
            wattn.to(device),
        )
        # Note: freq_scores is not passed to model.forward()
        # The forward() method doesn't accept it
        with torch.no_grad():
            for _ in range(n_warmup):
                model(
                    input_ids=iids,
                    attention_mask=amask,
                    words_lengths=wlen,
                    word_attention_mask=wattn,
                )
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            timings = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                model(
                    input_ids=iids,
                    attention_mask=amask,
                    words_lengths=wlen,
                    word_attention_mask=wattn,
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - t0)
        timings = np.asarray(timings)
        mean_latency_s = float(timings.mean())
        results[f"latency_ms_bs{bs}"] = mean_latency_s * 1000.0
        results[f"throughput_bs{bs}"] = bs / mean_latency_s if mean_latency_s > 0 else float("nan")
        results[f"peak_mem_MB_bs{bs}"] = (
            torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else float("nan")
        )
    return results


def run_latency_benchmark(base_args, bundle: Dict, trained_checkpoint_dir: str) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained_model(base_args, bundle, trained_checkpoint_dir, device)
    results = benchmark_latency_throughput(base_args, bundle, model, device, batch_sizes=(1, 8, 32))
    out_dir = os.path.join(base_args.output_dir, "latency")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "latency_benchmark.json"), "w") as f:
        json.dump(results, f, indent=2)
    del model
    _cleanup_cuda()
    return results


def _fig_dir(output_dir: str) -> str:
    d = os.path.join(output_dir, "figures")
    os.makedirs(d, exist_ok=True)
    return d


def fig_performance_vs_unfreeze(
    ratio_df: "pd.DataFrame", output_dir: str, metric: str = "mean_f1"
) -> Optional[str]:
    if ratio_df is None or ratio_df.empty:
        return None
    df = (
        ratio_df[~ratio_df.get("failed", False).astype(bool)]
        if "failed" in ratio_df.columns
        else ratio_df
    )
    df = df.dropna(subset=[metric, "unfrozen_ratio"])
    if df.empty:
        return None
    agg = df.groupby("unfrozen_ratio")[metric].agg(["mean", "std", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        agg["unfrozen_ratio"],
        agg["mean"],
        yerr=agg["std"].fillna(0.0),
        fmt="o-",
        capsize=4,
        color="steelblue",
        label=f"{metric} (mean +/- std)",
    )
    if len(agg) >= 3:
        coeffs = np.polyfit(agg["unfrozen_ratio"], agg["mean"], deg=2)
        xs = np.linspace(agg["unfrozen_ratio"].min(), agg["unfrozen_ratio"].max(), 100)
        ax.plot(xs, np.polyval(coeffs, xs), "--", color="crimson", alpha=0.7, label="quadratic fit")
    ax.set_xlabel("unfrozen_ratio (fraction of backbone layers trainable)")
    ax.set_ylabel(metric)
    ax.set_title("Performance vs. Unfrozen Backbone Ratio")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "01_performance_vs_unfreeze.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_layer_contribution_heatmap(contrib_df: "pd.DataFrame", output_dir: str) -> Optional[str]:
    if contrib_df is None or contrib_df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, max(4, 0.35 * len(contrib_df))))
    for ax, col, title, cmap in zip(
        axes,
        ["cka_vs_frozen", "grad_norm_mean"],
        [
            "Linear CKA vs. fully-frozen model\n(low = layer changed most)",
            "Mean gradient norm per layer\n(training-time signal strength)",
        ],
        ["viridis", "magma"],
    ):
        vals = contrib_df[col].to_numpy().reshape(-1, 1)
        im = ax.imshow(vals, aspect="auto", cmap=cmap)
        ax.set_yticks(range(len(contrib_df)))
        ax.set_yticklabels(contrib_df["layer"])
        ax.set_xticks([])
        ax.set_ylabel("backbone layer index")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.05)
    fig.suptitle("Layer-wise Contribution Analysis")
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "02_layer_contribution_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_pareto_frontier(
    df: "pd.DataFrame",
    output_dir: str,
    group_col: str,
    metric: str = "mean_f1",
    cost_col: str = "peak_gpu_mem_MB",
) -> Optional[str]:
    if df is None or df.empty or cost_col not in df.columns:
        return None
    clean = df.dropna(subset=[metric, cost_col])
    if clean.empty:
        return None
    agg = clean.groupby(group_col).agg(perf=(metric, "mean"), cost=(cost_col, "mean")).reset_index()
    pts = agg[["cost", "perf"]].to_numpy()
    order = np.argsort(-pts[:, 1])
    pareto_mask = np.zeros(len(pts), dtype=bool)
    best_cost_so_far = np.inf
    for idx in order:
        if pts[idx, 0] <= best_cost_so_far:
            pareto_mask[idx] = True
            best_cost_so_far = pts[idx, 0]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        agg.loc[~pareto_mask, "cost"],
        agg.loc[~pareto_mask, "perf"],
        c="gray",
        alpha=0.6,
        label="dominated",
    )
    ax.scatter(
        agg.loc[pareto_mask, "cost"],
        agg.loc[pareto_mask, "perf"],
        c="crimson",
        s=80,
        label="Pareto-optimal",
        zorder=3,
    )
    pf = agg.loc[pareto_mask].sort_values("cost")
    ax.plot(pf["cost"], pf["perf"], "--", c="crimson", alpha=0.7)
    for _, row in agg.iterrows():
        ax.annotate(
            str(row[group_col]),
            (row["cost"], row["perf"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set_xlabel(cost_col)
    ax.set_ylabel(metric)
    ax.set_title(f"Pareto Front: {metric} vs. {cost_col}")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "03_pareto_frontier.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_sensitivity_heatmap(sens_df: "pd.DataFrame", output_dir: str) -> Optional[str]:
    if sens_df is None or sens_df.empty:
        return None
    params = list(sens_df["param"].unique())
    n_pos = max((len(sens_df[sens_df.param == p]) for p in params))
    grid = np.full((len(params), n_pos), np.nan)
    for i, p in enumerate(params):
        vals = sens_df.loc[sens_df.param == p].sort_values("value")["score"].to_numpy()
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        lo, hi = (finite.min(), finite.max())
        norm = (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)
        grid[i, : len(norm)] = norm
    fig, ax = plt.subplots(figsize=(1.1 * n_pos + 3, 0.5 * len(params) + 2))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params)
    ax.set_xticks(range(n_pos))
    ax.set_xticklabels([f"v{j + 1}" for j in range(n_pos)])
    ax.set_xlabel("Grid position (low -> high value)")
    ax.set_title("Hyperparameter Sensitivity (normalized dev tuning metric)")
    fig.colorbar(im, ax=ax, label="normalized score")
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "04_sensitivity_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_ablation_barchart(
    ablation_df: "pd.DataFrame",
    sig_df: Optional["pd.DataFrame"],
    output_dir: str,
    metric: str = "mean_f1",
) -> Optional[str]:
    if ablation_df is None or ablation_df.empty:
        return None
    df = ablation_df.dropna(subset=[metric])
    agg = (
        df.groupby("experiment")[metric]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    stars = {}
    if (
        sig_df is not None
        and (not sig_df.empty)
        and ("t_pvalue_fdr_reject_at_0.05" in sig_df.columns)
    ):
        for _, row in sig_df.iterrows():
            stars[row["experiment"]] = "*" if row.get("t_pvalue_fdr_reject_at_0.05") else ""
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(agg)), 6))
    bars = ax.bar(
        agg["experiment"],
        agg["mean"],
        yerr=agg["std"].fillna(0.0),
        capsize=4,
        color="steelblue",
        edgecolor="black",
    )
    for bar, exp in zip(bars, agg["experiment"]):
        s = stars.get(exp, "")
        if s:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                s,
                ha="center",
                fontsize=14,
                color="crimson",
            )
    ax.set_ylabel(metric)
    ax.set_title(
        "Component Ablation (mean +/- std across seeds; * = FDR-significant vs. E0_full_model)"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "05_ablation_barchart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_learning_curves(history: Dict[str, List[float]], output_dir: str) -> Optional[str]:
    if not history or not history.get("epoch"):
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history["epoch"], history["train_loss"], label="train loss", marker="o")
    ax1.plot(history["epoch"], history["dev_loss"], label="dev loss", marker="s")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.set_title("Loss")
    ax2.plot(
        history["epoch"],
        history["dev_tuning_metric"],
        label="dev tuning metric",
        marker="^",
        color="darkorange",
    )
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("tuning metric")
    ax2.legend()
    ax2.set_title("Dev Tuning Metric")
    fig.suptitle("Learning Curves")
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "06_learning_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_exit_distribution(
    exit_layers: List[int], freq_scores: List[float], output_dir: str
) -> Optional[str]:
    if not exit_layers:
        return None
    el = np.array(exit_layers, dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(
        el,
        bins=int(el.max() - el.min() + 1) if el.max() > el.min() else 1,
        color="teal",
        edgecolor="black",
    )
    ax1.set_xlabel("exit layer")
    ax1.set_ylabel("count")
    ax1.set_title("Exit Layer Distribution")
    if freq_scores and len(freq_scores) == len(exit_layers):
        fs = np.array(freq_scores, dtype=float)
        r = float(np.corrcoef(fs, el)[0, 1]) if len(fs) > 2 else float("nan")
        ax2.scatter(fs, el, alpha=0.3, s=10, c="teal")
        if len(fs) > 2 and np.isfinite(r):
            m, b = np.polyfit(fs, el, 1)
            xs = np.linspace(fs.min(), fs.max(), 50)
            ax2.plot(xs, m * xs + b, "r--", label=f"linear fit (Pearson r={r:.3f})")
            ax2.legend()
        ax2.set_xlabel("word-rarity score")
        ax2.set_ylabel("exit layer")
        ax2.set_title("Rarity vs. Exit Layer")
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "07_exit_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_significance_table(sig_df: "pd.DataFrame", output_dir: str) -> Optional[str]:
    if sig_df is None or sig_df.empty:
        return None
    cols = [
        c
        for c in (
            "experiment",
            "unfrozen_ratio",
            "unfreeze_position",
            "n",
            "mean",
            "std",
            "delta",
            "t_pvalue",
            "cohens_d",
            "t_pvalue_fdr_reject_at_0.05",
        )
        if c in sig_df.columns
    ]
    disp = sig_df[cols].copy()
    for c in disp.columns:
        if disp[c].dtype == float:
            disp[c] = disp[c].map(lambda x: f"{x:.4g}" if pd.notna(x) else "NA")
    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 2, 0.4 * len(disp) + 2))
    ax.axis("off")
    tbl = ax.table(cellText=disp.values.tolist(), colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    ax.set_title("Statistical Significance vs. Baseline", pad=20)
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "08_significance_table.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_confusion_matrix(
    intent_true: Optional[np.ndarray],
    intent_pred: Optional[np.ndarray],
    intent_label_set: List[str],
    output_dir: str,
    max_labels: int = 30,
) -> Optional[str]:
    if intent_true is None or intent_pred is None or len(intent_true) == 0:
        logger.warning(
            "fig_confusion_matrix: no single-intent-per-utterance predictions captured (dataset may be predominantly multi-label/compound-intent, in which case a single N x N confusion matrix isn't a well-defined evaluation view); skipping."
        )
        return None
    n_lbl = len(intent_label_set)
    mat = np.zeros((n_lbl, n_lbl), dtype=int)
    for t, p in zip(intent_true, intent_pred):
        mat[int(t), int(p)] += 1
    if n_lbl > max_labels:
        freq = mat.sum(axis=1)
        keep = np.argsort(-freq)[:max_labels]
        mat = mat[np.ix_(keep, keep)]
        labels = [intent_label_set[i] for i in keep]
    else:
        labels = intent_label_set
    fig, ax = plt.subplots(figsize=(0.35 * len(labels) + 4, 0.35 * len(labels) + 4))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("Gold intent")
    ax.set_title(
        "Intent Confusion Matrix" + (" (top-frequency subset)" if n_lbl > max_labels else "")
    )
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "09_confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_architecture_overview(output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)

    def box(x, y, w, h, text, color="#dbe9f6"):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, wrap=True)

    box(0.3, 2.2, 1.6, 1.6, "Token\nEmbeddings\n(frozen)")
    for i in range(4):
        color = "#c6e6c6" if i < 2 else "#dbe9f6"
        box(
            2.2 + i * 1.5,
            2.2,
            1.2,
            1.6,
            f"Backbone\nLayer {i + 1}" + ("\n(...)" if i == 3 else ""),
            color=color,
        )
    box(8.3, 2.2, 1.4, 1.6, "Final\nHidden States")
    for i in range(4):
        box(2.2 + i * 1.5, 0.2, 1.2, 1.2, "PABEE\nintent/slot\nprobe head", color="#f6e0b5")
    box(
        0.3,
        4.4,
        2.1,
        1.2,
        "Word-Frequency Index\n(rarity -> per-sample\nmin exit layer)",
        color="#f6b5b5",
    )
    box(
        2.7,
        4.4,
        2.1,
        1.2,
        "Partial Unfreeze\n(unfrozen_ratio,\nunfreeze_position)",
        color="#c6e6c6",
    )
    box(5.1, 4.4, 2.1, 1.2, "Depth-Adaptive Patience\n(ee_patience_decay)", color="#f6e0b5")
    box(7.4, 4.4, 2.1, 1.2, "Exit Logit Smoothing\n(2-layer free ensemble)", color="#f6e0b5")
    ax.set_title("Joint Intent/Slot PABEE Architecture: Novelty Attachment Points (schematic)")
    fig.tight_layout()
    path = os.path.join(_fig_dir(output_dir), "10_architecture.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_all_figures(
    output_dir: str,
    ratio_df=None,
    position_df=None,
    contrib_df=None,
    ablation_df=None,
    ablation_sig_df=None,
    sens_df=None,
    history=None,
    exit_layers=None,
    freq_scores=None,
    intent_true=None,
    intent_pred=None,
    intent_label_set=None,
    metric: str = "mean_f1",
) -> Dict[str, Optional[str]]:
    paths: Dict[str, Optional[str]] = {}
    paths["performance_vs_unfreeze"] = (
        fig_performance_vs_unfreeze(ratio_df, output_dir, metric) if ratio_df is not None else None
    )
    paths["layer_contribution"] = (
        fig_layer_contribution_heatmap(contrib_df, output_dir) if contrib_df is not None else None
    )
    pareto_df = (
        ablation_df
        if ablation_df is not None and (not (ratio_df is not None and (not ratio_df.empty)))
        else ratio_df
    )
    pareto_group = "experiment" if pareto_df is ablation_df else "unfrozen_ratio"
    paths["pareto_frontier"] = (
        fig_pareto_frontier(pareto_df, output_dir, pareto_group, metric)
        if pareto_df is not None
        else None
    )
    paths["sensitivity_heatmap"] = (
        fig_sensitivity_heatmap(sens_df, output_dir) if sens_df is not None else None
    )
    paths["ablation_barchart"] = (
        fig_ablation_barchart(ablation_df, ablation_sig_df, output_dir, metric)
        if ablation_df is not None
        else None
    )
    paths["learning_curves"] = (
        fig_learning_curves(history, output_dir) if history is not None else None
    )
    paths["exit_distribution"] = (
        fig_exit_distribution(exit_layers, freq_scores, output_dir)
        if exit_layers is not None
        else None
    )
    paths["significance_table"] = (
        fig_significance_table(ablation_sig_df, output_dir) if ablation_sig_df is not None else None
    )
    paths["confusion_matrix"] = (
        fig_confusion_matrix(intent_true, intent_pred, intent_label_set, output_dir)
        if intent_label_set is not None
        else None
    )
    paths["architecture"] = fig_architecture_overview(output_dir)
    return paths


def run_final_train_eval_test(base_args, bundle: Dict, best_params: Optional[Dict] = None):
    final_args = _clone_args(base_args, **best_params or {})
    final_args.output_dir = os.path.join(base_args.output_dir, "final_model")
    final_args.use_wandb = False
    final_args.min_unfrozen_ratio_floor = (
        final_args.min_unfrozen_ratio_floor
        if hasattr(final_args, "min_unfrozen_ratio_floor")
        else 0.0
    )
    os.makedirs(final_args.output_dir, exist_ok=True)
    set_seed(getattr(base_args, "seed", 42))
    trainer = _new_trainer(final_args, bundle, use_full_train=True)
    trainer.train()
    trainer.load_model()
    test_results = trainer.evaluate("test")
    with open(os.path.join(final_args.output_dir, "best_hp_used.json"), "w") as f:
        json.dump(best_params or {}, f, indent=2)
    with open(os.path.join(final_args.output_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    return (trainer, test_results)


def run_full_pipeline(args) -> Dict:
    summary: Dict = {"stages_completed": [], "stages_failed": []}
    os.makedirs(args.output_dir, exist_ok=True)
    bundle = build_datasets_bundle(args)
    seeds = tuple(getattr(args, "seeds", [42, 43, 44]))
    metric = args.tuning_metric
    best_params: Optional[Dict] = None
    if getattr(args, "run_hpo", True) and _OPTUNA_AVAILABLE:
        try:
            best_params = run_hpo(args, bundle)
            summary["stages_completed"].append("hpo")
        except Exception as e:
            logger.error("HPO stage failed: %s", e)
            summary["stages_failed"].append({"stage": "hpo", "error": str(e)})
    best_hp_args = _clone_args(args, **best_params) if best_params else _clone_args(args)
    summary["best_params"] = best_params or {}
    final_trainer, test_results, history = (None, {}, {})
    try:
        final_trainer, test_results = run_final_train_eval_test(
            best_hp_args, bundle, best_params=None
        )
        history = final_trainer.history
        summary["stages_completed"].append("final_train_eval_test")
        summary["final_test_results"] = test_results
    except Exception as e:
        logger.error("Final train/eval/test stage failed: %s", e)
        summary["stages_failed"].append({"stage": "final_train_eval_test", "error": str(e)})
    final_checkpoint_dir = os.path.join(args.output_dir, "final_model")
    ratio_df, position_df = (pd.DataFrame(), pd.DataFrame())
    try:
        ratio_df = run_unfreeze_ratio_sweep(best_hp_args, bundle, seeds=seeds)
        position_df = run_unfreeze_position_sweep(best_hp_args, bundle, seeds=seeds)
        summary["stages_completed"].append("unfreeze_sweeps")
    except Exception as e:
        logger.error("Unfreeze sweep stage failed: %s", e)
        summary["stages_failed"].append({"stage": "unfreeze_sweeps", "error": str(e)})
    contrib = {}
    try:
        if not ratio_df.empty:
            clean = ratio_df[~ratio_df.get("failed", False).astype(bool)]
            best_ratio = float(clean.groupby("unfrozen_ratio")[metric].mean().idxmax())
        else:
            best_ratio = getattr(best_hp_args, "unfrozen_ratio", 0.5)
        contrib = run_layer_contribution_analysis(
            best_hp_args,
            bundle,
            best_ratio=best_ratio,
            trained_checkpoint_dir=final_checkpoint_dir,
            best_position="front",
        )
        summary["stages_completed"].append("layer_contribution")
    except Exception as e:
        logger.error("Layer contribution stage failed: %s", e)
        summary["stages_failed"].append({"stage": "layer_contribution", "error": str(e)})
    ablation_df, baseline_df = (pd.DataFrame(), pd.DataFrame())
    try:
        ablation_df = run_ablation_study(best_hp_args, bundle, seeds=seeds, best_params=best_params)
        summary["stages_completed"].append("ablation")
    except Exception as e:
        logger.error("Ablation stage failed: %s", e)
        summary["stages_failed"].append({"stage": "ablation", "error": str(e)})
    try:
        baseline_df = run_baseline_study(best_hp_args, bundle, seeds=seeds, best_params=best_params)
        summary["stages_completed"].append("baselines")
    except Exception as e:
        logger.error("Baseline stage failed: %s", e)
        summary["stages_failed"].append({"stage": "baselines", "error": str(e)})
    stats_dir = os.path.join(args.output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    ablation_sig_df, unfreeze_sig_df = (pd.DataFrame(), pd.DataFrame())
    try:
        if not ablation_df.empty:
            ablation_sig_df = run_statistical_tests(
                ablation_df, "experiment", "E0_full_model", metric
            )
            ablation_sig_df.to_csv(
                os.path.join(stats_dir, "statistical_significance.csv"), index=False
            )
            ablation_sig_df[["experiment", "cohens_d"]].to_csv(
                os.path.join(stats_dir, "effect_sizes.csv"), index=False
            )
        if not ratio_df.empty:
            unfreeze_sig_df = run_statistical_tests(ratio_df, "unfrozen_ratio", 0.5, metric)
            unfreeze_sig_df.to_csv(
                os.path.join(stats_dir, "unfreeze_ratio_significance.csv"), index=False
            )
        summary["stages_completed"].append("statistical_tests")
    except Exception as e:
        logger.error("Statistical testing stage failed: %s", e)
        summary["stages_failed"].append({"stage": "statistical_tests", "error": str(e)})
    sens_df, corr_df = (pd.DataFrame(), pd.DataFrame())
    try:
        sens_df, corr_df = run_sensitivity_analysis(best_hp_args, bundle, center_params=None)
        summary["stages_completed"].append("sensitivity")
    except Exception as e:
        logger.error("Sensitivity analysis stage failed: %s", e)
        summary["stages_failed"].append({"stage": "sensitivity", "error": str(e)})
    pareto_df = pd.DataFrame()
    try:
        pareto_df = run_pareto_operating_points(
            best_hp_args, bundle, trained_checkpoint_dir=final_checkpoint_dir
        )
        summary["stages_completed"].append("pareto")
    except Exception as e:
        logger.error("Pareto sweep stage failed: %s", e)
        summary["stages_failed"].append({"stage": "pareto", "error": str(e)})
    try:
        latency_results = run_latency_benchmark(
            best_hp_args, bundle, trained_checkpoint_dir=final_checkpoint_dir
        )
        summary["latency_benchmark"] = latency_results
        summary["stages_completed"].append("latency")
    except Exception as e:
        logger.error("Latency benchmark stage failed: %s", e)
        summary["stages_failed"].append({"stage": "latency", "error": str(e)})
    try:
        if final_trainer is not None:
            rarity_result = run_rarity_validation(best_hp_args, bundle, final_trainer)
            summary["rarity_validation"] = rarity_result["stats"]
            summary["stages_completed"].append("rarity_validation")
    except Exception as e:
        logger.error("Rarity validation stage failed: %s", e)
        summary["stages_failed"].append({"stage": "rarity_validation", "error": str(e)})
    try:
        exit_layers = final_trainer.last_exit_layers if final_trainer is not None else []
        freq_scores = final_trainer.last_freq_scores if final_trainer is not None else []
        intent_true = final_trainer.last_intent_true if final_trainer is not None else None
        intent_pred = final_trainer.last_intent_pred if final_trainer is not None else None
        fig_paths = generate_all_figures(
            args.output_dir,
            ratio_df=ratio_df,
            position_df=position_df,
            contrib_df=contrib.get("layer_contribution_df") if contrib else None,
            ablation_df=ablation_df,
            ablation_sig_df=ablation_sig_df,
            sens_df=sens_df,
            history=history,
            exit_layers=exit_layers,
            freq_scores=freq_scores,
            intent_true=intent_true,
            intent_pred=intent_pred,
            intent_label_set=bundle["intent_label_set"],
            metric=metric,
        )
        summary["figures"] = fig_paths
        summary["stages_completed"].append("figures")
    except Exception as e:
        logger.error("Figure generation stage failed: %s", e)
        summary["stages_failed"].append({"stage": "figures", "error": str(e)})
    with open(os.path.join(args.output_dir, "pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(
        "Pipeline complete. Stages completed: %s | failed: %s",
        summary["stages_completed"],
        [s["stage"] for s in summary["stages_failed"]],
    )
    return summary


def _add_v5_arguments(p):
    p.add_argument(
        "--mode",
        default="train",
        choices=[
            "train",
            "hpo",
            "unfreeze_ratio",
            "unfreeze_position",
            "layer_contribution",
            "ablation",
            "baselines",
            "sensitivity",
            "pareto",
            "latency",
            "stats",
            "full_pipeline",
        ],
        help="'train' = ORIGINAL V4 behaviour (--do_train/--do_eval), fully backward compatible. All other modes are V5 additions.",
    )
    p.add_argument(
        "--seeds",
        default="42,43,44",
        type=lambda s: [int(x) for x in s.split(",") if x.strip() != ""],
        help="Comma-separated seed list for unfreeze/ablation sweeps.",
    )
    p.add_argument("--run_hpo", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--n_trials", default=20, type=int, help="Optuna trial budget.")
    p.add_argument("--hpo_epochs", default=4, type=int)
    p.add_argument("--hpo_early_stopping", default=2, type=int)
    p.add_argument("--hpo_seed", default=42, type=int)
    p.add_argument("--study_name", default="intent_v5_hpo")
    p.add_argument(
        "--hpo_subset_fraction",
        default=1.0,
        type=float,
        help="Fraction of TRAIN used per HPO trial (dev/test never subsampled).",
    )
    p.add_argument("--baseline_exp", default="E0_full_model")
    p.add_argument("--sensitivity_epochs", default=3, type=int)
    p.add_argument("--sensitivity_early_stopping", default=2, type=int)
    p.add_argument(
        "--best_hp_json",
        default="",
        help="Path to a best_hyperparameters.json from a prior --mode hpo run, used to center --mode sensitivity / ablation / unfreeze sweeps.",
    )
    p.add_argument(
        "--results_csv",
        default="",
        help="For --mode stats: path to an existing ablation_results.csv or layer_ratio_results.csv to recompute significance tests from.",
    )
    p.add_argument(
        "--group_col",
        default="experiment",
        help="For --mode stats: column to group by (e.g. 'experiment' or 'unfrozen_ratio').",
    )
    p.add_argument(
        "--trained_checkpoint_dir",
        default="",
        help="Directory containing a trained checkpoint.pth, required for --mode layer_contribution / pareto / latency.",
    )
    return p


def _load_best_params(args) -> Optional[Dict]:
    if getattr(args, "best_hp_json", ""):
        with open(args.best_hp_json) as f:
            payload = json.load(f)
        return payload.get("best_params_translated", payload.get("best_params"))
    return None


if __name__ == "__main__":
    p = build_argparser()
    _add_v5_arguments(p)
    args = p.parse_args()
    args.exit_logit_smoothing = not args.disable_exit_logit_smoothing
    if args.mode == "train":
        if not args.do_train and (not args.do_eval):
            p.error("--mode train requires --do_train and/or --do_eval.")
        main(args)
    elif args.mode == "hpo":
        if args.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
        bundle = build_datasets_bundle(args)
        run_hpo(args, bundle)
    elif args.mode == "unfreeze_ratio":
        if args.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
        bundle = build_datasets_bundle(args)
        run_unfreeze_ratio_sweep(args, bundle, seeds=tuple(args.seeds))
    elif args.mode == "unfreeze_position":
        if args.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
        bundle = build_datasets_bundle(args)
        run_unfreeze_position_sweep(args, bundle, seeds=tuple(args.seeds))
    elif args.mode == "layer_contribution":
        if args.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
        if not args.trained_checkpoint_dir:
            p.error("--mode layer_contribution requires --trained_checkpoint_dir.")
        bundle = build_datasets_bundle(args)
        run_layer_contribution_analysis(
            args,
            bundle,
            best_ratio=getattr(args, "unfrozen_ratio", 0.5),
            trained_checkpoint_dir=args.trained_checkpoint_dir,
            best_position=getattr(args, "unfreeze_position", "front"),
        )
    elif args.mode == "ablation":
        if args.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
        bundle = build_datasets_bundle(args)
        best_params = _load_best_params(args)
        df = run_ablation_study(args, bundle, seeds=tuple(args.seeds), best_params=best_params)
        if not df.empty:
            sig_df = run_statistical_tests(df, "experiment", args.baseline_exp, args.tuning_metric)
            stats_dir = os.path.join(args.output_dir, "stats")
            os.makedirs(stats_dir, exist_ok=True)
            sig_df.to_csv(os.path.join(stats_dir, "statistical_significance.csv"), index=False)
    elif args.mode == "baselines":
        if args.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
        bundle = build_datasets_bundle(args)
        best_params = _load_best_params(args)
        df = run_baseline_study(args, bundle, seeds=tuple(args.seeds), best_params=best_params)
        if not df.empty:
            sig_df = run_statistical_tests(df, "experiment", "M4_full_method", args.tuning_metric)
            stats_dir = os.path.join(args.output_dir, "stats")
            os.makedirs(stats_dir, exist_ok=True)
            sig_df.to_csv(os.path.join(stats_dir, "baseline_significance.csv"), index=False)
    elif args.mode == "pareto":
        if args.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
        if not args.trained_checkpoint_dir:
            p.error("--mode pareto requires --trained_checkpoint_dir.")
        bundle = build_datasets_bundle(args)
        run_pareto_operating_points(
            args, bundle, trained_checkpoint_dir=args.trained_checkpoint_dir
        )
    elif args.mode == "latency":
        if args.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
        if not args.trained_checkpoint_dir:
            p.error("--mode latency requires --trained_checkpoint_dir.")
        bundle = build_datasets_bundle(args)
        run_latency_benchmark(args, bundle, trained_checkpoint_dir=args.trained_checkpoint_dir)
    elif args.mode == "sensitivity":
        if args.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
        bundle = build_datasets_bundle(args)
        best_params = _load_best_params(args)
        run_sensitivity_analysis(args, bundle, center_params=best_params)
    elif args.mode == "stats":
        if pd is None or not args.results_csv:
            p.error("--mode stats requires pandas and --results_csv.")
        df = pd.read_csv(args.results_csv)
        sig_df = run_statistical_tests(df, args.group_col, args.baseline_exp, args.tuning_metric)
        stats_dir = os.path.join(args.output_dir, "stats")
        os.makedirs(stats_dir, exist_ok=True)
        sig_df.to_csv(os.path.join(stats_dir, "statistical_significance.csv"), index=False)
        logger.info("Wrote %s", os.path.join(stats_dir, "statistical_significance.csv"))
    elif args.mode == "full_pipeline":
        if args.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
        run_full_pipeline(args)