#!/bin/bash

# ============================================================

# CONTINUE Intent-base-hpo.py AFTER final_model

#

# CURRENT EXPERIMENT

# ============================================================

# Dataset:

# chirunder/MixAtis_for_DecoderOnly

#

# Model:

# Qwen/Qwen2.5-1.5B

#

# Output:

# /home/tahir/AAAI-26-INTENT/output-base-hpo-mixatis-Qwen-1.5B

#

# Existing HPO:

# /home/tahir/AAAI-26-INTENT/output-base-hpo-mixatis-Qwen-1.5B/hpo/best_hyperparameters.json

#

# Already completed:

# - HPO

# - final_model

#

# This script continues with:

# 1. Unfreeze-ratio sweep

# 2. Unfreeze-position sweep

# 3. Layer contribution analysis

# 4. Component ablation

# 5. Statistical significance/effect sizes

# 6. Hyperparameter sensitivity

# 7. Figure generation

#

# IMPORTANT:

# ALL remaining experiments use the EXISTING HPO BEST

# PARAMETERS as their baseline.

#

# Each analysis changes only what it is designed to test.

#

# NO HPO rerun.

# NO final_model retraining.

#

# ============================================================

set -euo pipefail

# ============================================================

# 1. CONFIGURATION

# ============================================================

PROJECT_DIR="/home/tahir/AAAI-26-INTENT"

PYTHON_SCRIPT="$PROJECT_DIR/Intent-base-hpo.py"

DATASET="chirunder/MixAtis_for_DecoderOnly"

MODEL_NAME="Qwen/Qwen2.5-1.5B"

OUTPUT_DIR="/home/tahir/AAAI-26-INTENT/output-base-hpo-mixatis-Qwen-1.5B"

BEST_HP_JSON="$OUTPUT_DIR/hpo/best_hyperparameters.json"

FINAL_MODEL_DIR="$OUTPUT_DIR/final_model"

GPU=0

TUNING_METRIC="mean_f1"

MAX_SEQ_LENGTH=100

TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=4

SEEDS="42,43,44"

SENSITIVITY_EPOCHS=3
SENSITIVITY_EARLY_STOPPING=2

# ============================================================

# 2. ENVIRONMENT

# ============================================================

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"

cd "$PROJECT_DIR"

# ============================================================

# 3. VALIDATE FILES

# ============================================================

echo
echo "============================================================"
echo "VALIDATING CONTINUATION EXPERIMENT"
echo "============================================================"

echo "Python:"
echo "  $PYTHON_SCRIPT"

echo "Dataset:"
echo "  $DATASET"

echo "Model:"
echo "  $MODEL_NAME"

echo "Output:"
echo "  $OUTPUT_DIR"

echo "HPO:"
echo "  $BEST_HP_JSON"

echo "Final model:"
echo "  $FINAL_MODEL_DIR"

echo "GPU:"
echo "  $GPU"

echo "Seeds:"
echo "  $SEEDS"

echo "Metric:"
echo "  $TUNING_METRIC"

if [ ! -f "$PYTHON_SCRIPT" ]; then
echo
echo "ERROR: Python script does not exist:"
echo "  $PYTHON_SCRIPT"
exit 1
fi

if [ ! -f "$BEST_HP_JSON" ]; then
echo
echo "ERROR: HPO JSON does not exist:"
echo "  $BEST_HP_JSON"
exit 1
fi

if [ ! -d "$FINAL_MODEL_DIR" ]; then
echo
echo "ERROR: final_model directory does not exist:"
echo "  $FINAL_MODEL_DIR"
exit 1
fi

if [ ! -f "$FINAL_MODEL_DIR/checkpoint.pth" ]; then
echo
echo "ERROR: final_model checkpoint does not exist:"
echo "  $FINAL_MODEL_DIR/checkpoint.pth"
exit 1
fi

# ============================================================

# 4. CREATE OUTPUT DIRECTORIES

# ============================================================

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/unfreeze"
mkdir -p "$OUTPUT_DIR/ablation"
mkdir -p "$OUTPUT_DIR/stats"
mkdir -p "$OUTPUT_DIR/sensitivity"
mkdir -p "$OUTPUT_DIR/figures"

# ============================================================

# 5. DISPLAY HPO PARAMETERS

# ============================================================

echo
echo "============================================================"
echo "EXISTING BEST HYPERPARAMETERS"
echo "============================================================"

cat "$BEST_HP_JSON"

echo
echo "============================================================"
echo "STARTING POST-FINAL-MODEL PIPELINE"
echo "============================================================"

# ============================================================

# IMPORTANT:

#

# Use a QUOTED heredoc:

#

# <<'PY'

#

# This prevents Bash from interpreting Python constructs such

# as:

#

# $(...)

# ${...}

#

# as shell expressions.

#

# Configuration is passed safely through environment variables.

# ============================================================

export INTENT_PROJECT_DIR="$PROJECT_DIR"
export INTENT_PYTHON_SCRIPT="$PYTHON_SCRIPT"
export INTENT_DATASET="$DATASET"
export INTENT_MODEL_NAME="$MODEL_NAME"
export INTENT_OUTPUT_DIR="$OUTPUT_DIR"
export INTENT_BEST_HP_JSON="$BEST_HP_JSON"
export INTENT_FINAL_MODEL_DIR="$FINAL_MODEL_DIR"

export INTENT_GPU="$GPU"
export INTENT_TUNING_METRIC="$TUNING_METRIC"
export INTENT_MAX_SEQ_LENGTH="$MAX_SEQ_LENGTH"
export INTENT_TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE"
export INTENT_EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE"
export INTENT_SEEDS="$SEEDS"
export INTENT_SENSITIVITY_EPOCHS="$SENSITIVITY_EPOCHS"
export INTENT_SENSITIVITY_EARLY_STOPPING="$SENSITIVITY_EARLY_STOPPING"

python - <<'PY'

# ============================================================

# PYTHON CONTINUATION

# ============================================================

import os
import json
import gc
import importlib.util

import pandas as pd
import torch

# ============================================================

# READ CONFIGURATION SAFELY FROM ENVIRONMENT

# ============================================================

PROJECT_DIR = os.environ["INTENT_PROJECT_DIR"]

PYTHON_SCRIPT = os.environ["INTENT_PYTHON_SCRIPT"]

DATASET = os.environ["INTENT_DATASET"]

MODEL_NAME = os.environ["INTENT_MODEL_NAME"]

OUTPUT_DIR = os.environ["INTENT_OUTPUT_DIR"]

BEST_HP_JSON = os.environ["INTENT_BEST_HP_JSON"]

FINAL_MODEL_DIR = os.environ["INTENT_FINAL_MODEL_DIR"]

GPU = int(os.environ["INTENT_GPU"])

TUNING_METRIC = os.environ["INTENT_TUNING_METRIC"]

MAX_SEQ_LENGTH = int(
os.environ["INTENT_MAX_SEQ_LENGTH"]
)

TRAIN_BATCH_SIZE = int(
os.environ["INTENT_TRAIN_BATCH_SIZE"]
)

EVAL_BATCH_SIZE = int(
os.environ["INTENT_EVAL_BATCH_SIZE"]
)

SEEDS = tuple(
int(x.strip())
for x in os.environ["INTENT_SEEDS"].split(",")
if x.strip()
)

SENSITIVITY_EPOCHS = int(
os.environ["INTENT_SENSITIVITY_EPOCHS"]
)

SENSITIVITY_EARLY_STOPPING = int(
os.environ["INTENT_SENSITIVITY_EARLY_STOPPING"]
)

# ============================================================

# PRINT CONFIGURATION

# ============================================================

print()
print("============================================================")
print("POST-FINAL-MODEL CONTINUATION")
print("============================================================")

print("Dataset:")
print(" ", DATASET)

print("Model:")
print(" ", MODEL_NAME)

print("Output:")
print(" ", OUTPUT_DIR)

print("HPO:")
print(" ", BEST_HP_JSON)

print("Final model:")
print(" ", FINAL_MODEL_DIR)

print("GPU:")
print(" ", GPU)

print("Seeds:")
print(" ", SEEDS)

print("Metric:")
print(" ", TUNING_METRIC)

print("Max sequence length:")
print(" ", MAX_SEQ_LENGTH)

print("Train batch:")
print(" ", TRAIN_BATCH_SIZE)

print("Eval batch:")
print(" ", EVAL_BATCH_SIZE)

print("Sensitivity epochs:")
print(" ", SENSITIVITY_EPOCHS)

print("Sensitivity early stopping:")
print(" ", SENSITIVITY_EARLY_STOPPING)

# ============================================================

# IMPORT CURRENT Intent-base-hpo.py

# ============================================================

print()
print("============================================================")
print("IMPORTING CURRENT Intent-base-hpo.py")
print("============================================================")

spec = importlib.util.spec_from_file_location(
"intent_base_hpo_current",
PYTHON_SCRIPT
)

if spec is None:
raise RuntimeError(
f"Could not create import specification for {PYTHON_SCRIPT}"
)

if spec.loader is None:
raise RuntimeError(
f"Could not create loader for {PYTHON_SCRIPT}"
)

mod = importlib.util.module_from_spec(spec)

spec.loader.exec_module(mod)

print("Successfully imported:")
print(PYTHON_SCRIPT)

# ============================================================

# LOAD EXISTING HPO PARAMETERS

# ============================================================

print()
print("============================================================")
print("LOADING EXISTING HPO PARAMETERS")
print("============================================================")

with open(BEST_HP_JSON, "r") as f:
hpo_payload = json.load(f)

# The Python implementation writes:

#

# best_params_raw

# best_params_translated

#

# We must use translated because the source code explicitly

# translates Optuna names into the actual argparse names.

#

# Example:

#

# learning_rate_backbone

# ->

# backbone_learning_rate

#

# learning_rate_head

# ->

# learning_rate

#

# batch_size

# ->

# train_batch_size

#

# ============================================================

best_params = hpo_payload.get(
"best_params_translated"
)

if best_params is None:

```
# Backward-compatible fallback.
best_params = hpo_payload.get(
    "best_params"
)
```

if not best_params:

```
raise RuntimeError(
    "No HPO parameters found in:\n"
    f"{BEST_HP_JSON}\n\n"
    "Expected 'best_params_translated'."
)
```

print()
print("HPO baseline:")
print(
json.dumps(
best_params,
indent=2,
sort_keys=True
)
)

# ============================================================

# BUILD ARGUMENT PARSER FROM EXACT CURRENT PYTHON FILE

# ============================================================

parser = mod.build_argparser()

mod._add_v5_arguments(parser)

# ------------------------------------------------------------

# Construct initial args.

#

# These are experiment-level settings.

# HPO parameters will then be overlaid.

# ------------------------------------------------------------

base_args = parser.parse_args([
"--mode",
"unfreeze_ratio",

```
"--hf_dataset",
DATASET,

"--model_name_or_path",
MODEL_NAME,

"--output_dir",
OUTPUT_DIR,

"--gpu",
str(GPU),

"--tuning_metric",
TUNING_METRIC,

"--max_seq_length",
str(MAX_SEQ_LENGTH),

"--train_batch_size",
str(TRAIN_BATCH_SIZE),

"--eval_batch_size",
str(EVAL_BATCH_SIZE),

"--seeds",
",".join(
    str(x)
    for x in SEEDS
),

"--best_hp_json",
BEST_HP_JSON,

"--run_hpo",
"false",

"--sensitivity_epochs",
str(SENSITIVITY_EPOCHS),

"--sensitivity_early_stopping",
str(SENSITIVITY_EARLY_STOPPING),
```

])

# ============================================================

# CRITICAL:

# APPLY EXISTING HPO PARAMETERS TO BASE CONFIGURATION

# ============================================================

base_args = mod._clone_args(
base_args,
**best_params
)

# ============================================================

# REASSERT CURRENT EXPERIMENT IDENTITY

#

# These are not HPO search variables for this continuation.

# They define the current experiment itself.

# ============================================================

base_args.hf_dataset = DATASET

base_args.model_name_or_path = MODEL_NAME

base_args.output_dir = OUTPUT_DIR

base_args.gpu = GPU

base_args.tuning_metric = TUNING_METRIC

base_args.max_seq_length = MAX_SEQ_LENGTH

base_args.train_batch_size = TRAIN_BATCH_SIZE

base_args.eval_batch_size = EVAL_BATCH_SIZE

base_args.seeds = list(SEEDS)

base_args.best_hp_json = BEST_HP_JSON

base_args.run_hpo = False

base_args.sensitivity_epochs = SENSITIVITY_EPOCHS

base_args.sensitivity_early_stopping = (
SENSITIVITY_EARLY_STOPPING
)

# Needed because the ratio sweep explicitly includes 0.0.

base_args.min_unfrozen_ratio_floor = 0.0

# No W&B for continuation experiments.

base_args.use_wandb = False

# ============================================================

# DISPLAY EFFECTIVE BASELINE

# ============================================================

print()
print("============================================================")
print("EFFECTIVE HPO-BASED BASELINE")
print("============================================================")

important_keys = [
"hf_dataset",
"model_name_or_path",
"output_dir",
"gpu",
"tuning_metric",
"max_seq_length",
"train_batch_size",
"eval_batch_size",
"seeds",

```
# HPO / training configuration:
"learning_rate",
"backbone_learning_rate",
"gradient_accumulation_steps",

"unfrozen_ratio",
"unfreeze_position",

"use_freq_exit",
"intent_loss_fn",
"ee_patience_decay",
"exit_logit_smoothing",

"ee_patience",
"tau_slot",
"intent_exit_margin",
"loss_coef_intent",
"dropout_rate",
"weight_decay",
"min_exit_layer",

"best_hp_json",
"run_hpo",
```

]

for key in important_keys:

```
if hasattr(base_args, key):

    print(
        f"{key} = "
        f"{getattr(base_args, key)}"
    )
```

# ============================================================

# CUDA

# ============================================================

if torch.cuda.is_available():

```
torch.cuda.set_device(
    GPU
)
```

# ============================================================

# BUILD DATASET BUNDLE

# ============================================================

print()
print("============================================================")
print("BUILDING DATASET BUNDLE")
print("============================================================")

bundle = mod.build_datasets_bundle(
base_args
)

print("Dataset bundle successfully built.")

# ============================================================

# STAGE 1

# UNFREEZE RATIO SWEEP

# ============================================================

#

# The current Python implementation does:

#

# trial_args = clone(

# base_args,

# unfrozen_ratio=ratio,

# unfreeze_position="front",

# ...

# )

#

# Thus:

#

# HPO parameters

# +

# only ratio changed

#

# ============================================================

print()
print()
print("============================================================")
print("STAGE 1/7")
print("UNFREEZE RATIO SWEEP")
print("============================================================")

ratio_df = mod.run_unfreeze_ratio_sweep(
base_args,
bundle,
seeds=SEEDS
)

if ratio_df is None:

```
raise RuntimeError(
    "run_unfreeze_ratio_sweep returned None."
)
```

if ratio_df.empty:

```
raise RuntimeError(
    "Unfreeze ratio sweep produced zero rows."
)
```

ratio_csv = os.path.join(
OUTPUT_DIR,
"unfreeze",
"layer_ratio_results.csv"
)

print()
print("Unfreeze ratio sweep complete.")
print("Rows:", len(ratio_df))
print("Result:", ratio_csv)

# ============================================================

# SELECT BEST RATIO

# ============================================================

clean_ratio = ratio_df.copy()

if "failed" in clean_ratio.columns:

```
clean_ratio = clean_ratio[
    ~clean_ratio["failed"].astype(bool)
]
```

if clean_ratio.empty:

```
raise RuntimeError(
    "Every unfreeze-ratio experiment failed."
)
```

if TUNING_METRIC not in clean_ratio.columns:

```
raise RuntimeError(
    f"Metric '{TUNING_METRIC}' not found in "
    "ratio results.\n"
    f"Available columns: "
    f"{list(clean_ratio.columns)}"
)
```

ratio_means = (
clean_ratio
.groupby("unfrozen_ratio")[TUNING_METRIC]
.mean()
.sort_values(
ascending=False
)
)

BEST_RATIO = float(
ratio_means.index[0]
)

print()
print("============================================================")
print("BEST UNFREEZE RATIO")
print("============================================================")

print(
"Best ratio:",
BEST_RATIO
)

print(
"Best mean",
TUNING_METRIC,
":",
float(ratio_means.iloc[0])
)

print()
print(
ratio_means.to_string()
)

# ============================================================

# STAGE 2

# UNFREEZE POSITION SWEEP

# ============================================================

#

# Exact source behavior:

#

# unfrozen_ratio = 1.0

# unfreeze_position = position

#

# All other configuration comes from base_args,

# which already contains HPO parameters.

#

# ============================================================

print()
print()
print("============================================================")
print("STAGE 2/7")
print("UNFREEZE POSITION SWEEP")
print("============================================================")

position_df = mod.run_unfreeze_position_sweep(
base_args,
bundle,
seeds=SEEDS
)

if position_df is None:

```
raise RuntimeError(
    "run_unfreeze_position_sweep returned None."
)
```

if position_df.empty:

```
raise RuntimeError(
    "Unfreeze position sweep produced zero rows."
)
```

position_csv = os.path.join(
OUTPUT_DIR,
"unfreeze",
"layer_position_results.csv"
)

print()
print("Unfreeze position sweep complete.")
print("Rows:", len(position_df))
print("Result:", position_csv)

# ============================================================

# STAGE 3

# LAYER CONTRIBUTION ANALYSIS

# ============================================================

#

# Follow the current full-pipeline behavior:

#

# best_ratio = selected from ratio sweep

# best_position = "front"

#

# The analysis compares:

#

# - fully frozen reference

# - best partial-unfreeze configuration

#

# using:

#

# - CKA

# - gradient-flow analysis

#

# ============================================================

print()
print()
print("============================================================")
print("STAGE 3/7")
print("LAYER CONTRIBUTION ANALYSIS")
print("============================================================")

contrib = mod.run_layer_contribution_analysis(
base_args,
bundle,
best_ratio=BEST_RATIO,
best_position="front"
)

if contrib is None:

```
raise RuntimeError(
    "Layer contribution analysis returned None."
)
```

contrib_df = contrib.get(
"layer_contribution_df"
)

if contrib_df is None:

```
raise RuntimeError(
    "layer_contribution_df missing from "
    "layer contribution result."
)
```

if contrib_df.empty:

```
raise RuntimeError(
    "Layer contribution dataframe is empty."
)
```

contrib_csv = os.path.join(
OUTPUT_DIR,
"unfreeze",
"layer_contribution.csv"
)

print()
print("Layer contribution analysis complete.")
print("Result:", contrib_csv)

if "summary" in contrib:

```
print()
print("Layer contribution summary:")

print(
    json.dumps(
        contrib["summary"],
        indent=2,
        default=str
    )
)
```

# ============================================================

# STAGE 4

# COMPONENT ABLATION

# ============================================================

#

# IMPORTANT:

#

# The current source code calls:

#

# build_ablation_experiments(best_params)

#

# so E0-E5 are based on the HPO configuration.

#

# Only the intended ablated component changes.

#

# ============================================================

print()
print()
print("============================================================")
print("STAGE 4/7")
print("COMPONENT ABLATION")
print("============================================================")

ablation_df = mod.run_ablation_study(
base_args,
bundle,
seeds=SEEDS,
best_params=best_params
)

if ablation_df is None:

```
raise RuntimeError(
    "run_ablation_study returned None."
)
```

if ablation_df.empty:

```
raise RuntimeError(
    "Ablation study produced zero rows."
)
```

ablation_csv = os.path.join(
OUTPUT_DIR,
"ablation",
"ablation_results.csv"
)

print()
print("Ablation complete.")
print("Rows:", len(ablation_df))
print("Result:", ablation_csv)

# ============================================================

# STAGE 5

# STATISTICAL TESTS

# ============================================================

#

# Follow the statistical functionality of the source code.

#

# 5A:

# Ablation experiments vs E0_full_model

#

# 5B:

# Ratio sweep vs ratio=0.5

#

# No new training occurs in this stage.

#

# ============================================================

print()
print()
print("============================================================")
print("STAGE 5/7")
print("STATISTICAL SIGNIFICANCE + EFFECT SIZES")
print("============================================================")

stats_dir = os.path.join(
OUTPUT_DIR,
"stats"
)

os.makedirs(
stats_dir,
exist_ok=True
)

# ------------------------------------------------------------

# 5A. Ablation significance

# ------------------------------------------------------------

ablation_sig_df = mod.run_statistical_tests(
ablation_df,
"experiment",
"E0_full_model",
TUNING_METRIC
)

ablation_sig_csv = os.path.join(
stats_dir,
"statistical_significance.csv"
)

ablation_sig_df.to_csv(
ablation_sig_csv,
index=False
)

print(
"Ablation statistical results:",
ablation_sig_csv
)

# ------------------------------------------------------------

# 5B. Effect sizes

# ------------------------------------------------------------

effect_csv = os.path.join(
stats_dir,
"effect_sizes.csv"
)

if not ablation_sig_df.empty:

```
if "cohens_d" in ablation_sig_df.columns:

    cols = [
        "experiment",
        "cohens_d"
    ]

    cols = [
        c
        for c in cols
        if c in ablation_sig_df.columns
    ]

    ablation_sig_df[
        cols
    ].to_csv(
        effect_csv,
        index=False
    )

    print(
        "Effect sizes:",
        effect_csv
    )
```

# ------------------------------------------------------------

# 5C. Ratio statistical significance

#

# Preserve the source pipeline's ratio baseline:

#

# baseline = 0.5

#

# This is a statistical comparison only.

# It does NOT trigger additional training.

# ------------------------------------------------------------

ratio_sig_df = mod.run_statistical_tests(
ratio_df,
"unfrozen_ratio",
0.5,
TUNING_METRIC
)

ratio_sig_csv = os.path.join(
stats_dir,
"unfreeze_ratio_significance.csv"
)

ratio_sig_df.to_csv(
ratio_sig_csv,
index=False
)

print(
"Unfreeze-ratio statistical results:",
ratio_sig_csv
)

# ============================================================

# STAGE 6

# HYPERPARAMETER SENSITIVITY

# ============================================================

#

# This is explicitly HPO-centered.

#

# Source behavior:

#

# trial_args = clone(base_args, **center)

#

# center = best_params

#

# then:

#

# setattr(trial_args, param, val)

#

# Therefore each sensitivity experiment:

#

# starts from HPO optimum

# changes ONE parameter

#

# ============================================================

print()
print()
print("============================================================")
print("STAGE 6/7")
print("HYPERPARAMETER SENSITIVITY")
print("============================================================")

base_args.sensitivity_epochs = (
SENSITIVITY_EPOCHS
)

base_args.sensitivity_early_stopping = (
SENSITIVITY_EARLY_STOPPING
)

sens_df, corr_df = mod.run_sensitivity_analysis(
base_args,
bundle,
center_params=best_params
)

if sens_df is None:

```
raise RuntimeError(
    "Sensitivity analysis returned None for results."
)
```

if sens_df.empty:

```
raise RuntimeError(
    "Sensitivity analysis produced zero rows."
)
```

sens_csv = os.path.join(
OUTPUT_DIR,
"sensitivity",
"sensitivity_results.csv"
)

corr_csv = os.path.join(
OUTPUT_DIR,
"sensitivity",
"sensitivity_correlations.csv"
)

print()
print("Sensitivity analysis complete.")
print("Rows:", len(sens_df))
print("Results:", sens_csv)
print("Correlations:", corr_csv)

# ============================================================

# STAGE 7

# LOAD EXISTING final_model

# ============================================================

#

# We DO NOT retrain final_model.

#

# The model was already trained in the completed previous

# pipeline.

#

# We load checkpoint.pth only to recover the diagnostics

# required by generate_all_figures().

#

# ============================================================

print()
print()
print("============================================================")
print("STAGE 7/7")
print("LOADING EXISTING FINAL MODEL FOR FIGURES")
print("============================================================")

final_args = mod._clone_args(
base_args,
**best_params
)

final_args.output_dir = (
FINAL_MODEL_DIR
)

final_args.use_wandb = False

if torch.cuda.is_available():

```
torch.cuda.set_device(
    GPU
)
```

checkpoint_path = os.path.join(
FINAL_MODEL_DIR,
"checkpoint.pth"
)

print()
print("Existing checkpoint:")
print(checkpoint_path)

final_trainer = mod._new_trainer(
final_args,
bundle,
use_full_train=True
)

final_trainer.load_model()

# ============================================================

# EVALUATE EXISTING FINAL MODEL

# ============================================================

print()
print("Evaluating existing final model on test set.")

print("NO final-model training will occur.")

final_test_results = (
final_trainer.evaluate(
"test",
log_wandb=False,
quiet=True
)
)

print()
print("Existing final model evaluated.")

# ============================================================

# RECOVER DIAGNOSTICS

# ============================================================

exit_layers = (
final_trainer.last_exit_layers
)

freq_scores = (
final_trainer.last_freq_scores
)

intent_true = (
final_trainer.last_intent_true
)

intent_pred = (
final_trainer.last_intent_pred
)

intent_label_set = (
bundle["intent_label_set"]
)

# ============================================================

# HISTORY

# ============================================================

#

# The original final_model training occurred in a previous

# process, so its in-memory trainer.history is unavailable.

#

# Do NOT fabricate learning curves.

#

# generate_all_figures() accepts history=None.

# ============================================================

history = None

# ============================================================

# GENERATE FIGURES

# ============================================================

print()
print()
print("============================================================")
print("GENERATING FIGURES")
print("============================================================")

fig_paths = mod.generate_all_figures(
OUTPUT_DIR,

```
ratio_df=ratio_df,

position_df=position_df,

contrib_df=contrib_df,

ablation_df=ablation_df,

ablation_sig_df=ablation_sig_df,

sens_df=sens_df,

history=history,

exit_layers=exit_layers,

freq_scores=freq_scores,

intent_true=intent_true,

intent_pred=intent_pred,

intent_label_set=intent_label_set,

metric=TUNING_METRIC,
```

)

print()
print("============================================================")
print("FIGURES GENERATED")
print("============================================================")

for name, path in fig_paths.items():

```
print(
    f"{name:40s} -> {path}"
)
```

# ============================================================

# SAVE CONTINUATION SUMMARY

# ============================================================

summary = {

```
"continuation_after_final_model": True,

"hpo_rerun": False,

"final_model_retrained": False,

"dataset": DATASET,

"model_name_or_path": MODEL_NAME,

"output_dir": OUTPUT_DIR,

"best_hyperparameters_json": BEST_HP_JSON,

"best_hyperparameters": best_params,

"tuning_metric": TUNING_METRIC,

"max_seq_length": MAX_SEQ_LENGTH,

"train_batch_size": TRAIN_BATCH_SIZE,

"eval_batch_size": EVAL_BATCH_SIZE,

"seeds": list(SEEDS),

"sensitivity_epochs": SENSITIVITY_EPOCHS,

"sensitivity_early_stopping": SENSITIVITY_EARLY_STOPPING,

"best_unfrozen_ratio": BEST_RATIO,

"layer_contribution_position": "front",

"stages_completed": [

    "unfreeze_ratio",

    "unfreeze_position",

    "layer_contribution",

    "ablation",

    "statistical_tests",

    "sensitivity",

    "figure_generation",

],

"final_test_results": final_test_results,

"figures": fig_paths,
```

}

summary_path = os.path.join(
OUTPUT_DIR,
"continuation_pipeline_summary.json"
)

with open(
summary_path,
"w"
) as f:

```
json.dump(
    summary,
    f,
    indent=2,
    default=str
)
```

print()
print("============================================================")
print("POST-FINAL-MODEL PIPELINE COMPLETE")
print("============================================================")

print()
print("Summary:")
print(summary_path)

print()
print("Unfreeze:")
print(
os.path.join(
OUTPUT_DIR,
"unfreeze"
)
)

print()
print("Ablation:")
print(
os.path.join(
OUTPUT_DIR,
"ablation"
)
)

print()
print("Statistics:")
print(
os.path.join(
OUTPUT_DIR,
"stats"
)
)

print()
print("Sensitivity:")
print(
os.path.join(
OUTPUT_DIR,
"sensitivity"
)
)

print()
print("Figures:")
print(
os.path.join(
OUTPUT_DIR,
"figures"
)
)

print()
print("============================================================")
print("DONE")
print("============================================================")

# ============================================================

# CLEANUP

# ============================================================

del final_trainer

if torch.cuda.is_available():

```
torch.cuda.empty_cache()
```

gc.collect()

PY

# ============================================================

# SHELL COMPLETION

# ============================================================

echo
echo "============================================================"
echo "CONTINUATION SCRIPT FINISHED"
echo "============================================================"

echo
echo "Dataset:"
echo "  $DATASET"

echo
echo "Model:"
echo "  $MODEL_NAME"

echo
echo "Output:"
echo "  $OUTPUT_DIR"

echo
echo "HPO:"
echo "  $BEST_HP_JSON"

echo
echo "Results:"
echo "  $OUTPUT_DIR/unfreeze/"
echo "  $OUTPUT_DIR/ablation/"
echo "  $OUTPUT_DIR/stats/"
echo "  $OUTPUT_DIR/sensitivity/"
echo "  $OUTPUT_DIR/figures/"

echo
echo "Summary:"
echo "  $OUTPUT_DIR/continuation_pipeline_summary.json"

echo
echo "============================================================"
echo "DONE"
echo "============================================================"
