# ReViSQL: Achieving Human-Level Text-to-SQL

**SIGMOD 2027 Artifact**

> ReViSQL achieves human-level accuracy on BIRD for the first time without any multi-stage pipeline. It fine-tunes an LLM with reinforcement learning with verifiable rewards (RLVR) on BIRD-Platinum — a dataset of 2,462 expert-verified Text-to-SQL instances — and reaches human parity with inference-time scaling via majority voting.


## Framework Overview

ReViSQL reaches human parity through three procedures:

1. **BIRD-Platinum** — Expert-curated training data. We corrected annotation errors across 52.1% of SQL queries, 26.2% of questions, and 18.2% of external knowledge entries in 2,462 instances sampled from BIRD Train.

2. **RLVR Training** — Fine-tuning with CISPO. We train two configurations:
   - **ReViSQL** (the recipe): RLVR on BIRD-Platinum with a result-based reward that grades a rollout by whether its execution result matches the verified gold. We fine-tune Qwen3-235B-A22B with this recipe (**ReViSQL-235B**).
   - **ReViSQL-BIRD** (BIRD-specialized): adds two reward-shaping techniques to the result-based reward — a **VeriEQL** formal-equivalence check that softly down-weights rollouts whose results match by coincidence but are not provably equivalent to the gold, and an **evidence-supervision** reward that requires the model to translate and verify each external-knowledge entry. We fine-tune Kimi-K2.6 with this variant (**ReViSQL-BIRD-K2.6**), the model that reaches human parity. No process reward model (PRM) is used.

3. **Inference-time Scaling** — At inference, the fine-tuned model generates *N* candidate queries per question, and the final answer is selected by execution-based majority voting.


## Repository Structure

```
ReViSQL/
├── data/                          # Datasets (tracked in git; files keep the earlier `bird-verified-` prefix)
│   ├── bird-verified-train.json   # BIRD-Platinum training split (85%, 2,088 instances)
│   ├── bird-verified-val.json     # BIRD-Platinum validation split (15%, 374 instances)
│   ├── bird-verified-train.parquet        # Training split in parquet (output of curate_final_data.py)
│   ├── arcwise_plat_full.json     # Arcwise-Plat-Full eval set (498 questions, all errors corrected)
│   ├── arcwise_plat_sql.json      # Arcwise-Plat-SQL eval set (498 questions, SQL-only corrections)
│   ├── val_test.parquet           # Combined val+eval set in parquet (output of curate_final_data.py)
│   ├── bird-plat-2.5k-v1.json    # Raw BIRD-Platinum 2.5k training candidates (pre-verification)
│   ├── bird_eval.parquet          # Original BIRD mini-Dev evaluation set
│   ├── spider2sqlite_eval.parquet # Spider 2-SQLite evaluation set (135 questions)
│   ├── spider_2_snow_eval.parquet # Spider 2-Snow evaluation set (547 questions)
│   └── ddls/                      # Database schema DDL files (one per db_id)
│
├── scripts/
│   ├── train.py                   # RLVR fine-tuning with CISPO (via Tinker)
│   └── infer.py                   # Inference + evaluation with RLTestSetEvaluator
│
├── data_scripts/                  # Data preparation scripts
│   ├── curate_final_data.py       # Main curation: produces bird-verified-train.parquet + val_test.parquet
│   ├── curate_RL_dataset.py       # Earlier curation pipeline (intermediate artifacts)
│   ├── construct_db_schema.py     # DDL generation for BIRD databases
│   ├── construct_spider2sqlite_db_schema.py  # DDL generation for Spider 2-SQLite databases
│   └── generate_spider2sqlite_tables.py      # tables.json generation for Spider 2-SQLite
│
├── tinker_cookbook/               # Bundled training framework (RLVR engine, SQL env, evaluators)
├── pyproject.toml                 # Project dependencies
└── .gitignore
```

---

## Setup

**Requirements:** Python 3.11+.

```bash
# Install ReViSQL and dependencies
uv sync

# VeriEQL (formal SQL equivalence, used by the ReViSQL-BIRD training reward) is an optional extra:
uv sync --extra verieql

# Copy .env and fill in API keys
cp .env.example .env
```

**Required environment variables** (in `.env`):
```
TINKER_API_KEY=...        # For model training/sampling via Tinker
WANDB_API_KEY=...         # For training/eval logging
```

**Database files:** Download the BIRD mini-Dev databases and Spider 2-SQLite databases and note their paths. These are passed at runtime via `--db_path`.

---

## BIRD-Platinum Dataset

BIRD-Platinum is available in `data/`; the files keep the earlier `bird-verified-` prefix. The key files:

| File | Description |
|---|---|
| `data/bird-verified-train.json` | 2,088 training instances (85% split) |
| `data/bird-verified-val.json` | 374 validation instances (15% split) |
| `data/arcwise_plat_sql.json` | Arcwise-Plat-SQL evaluation set (498 questions, SQL corrected) |
| `data/arcwise_plat_full.json` | Arcwise-Plat-Full evaluation set (498 questions, fully corrected) |

To regenerate the parquet files used by the training scripts:
```bash
# Run from the repo root
uv run data_scripts/curate_final_data.py
# Outputs: data/bird-verified-train.parquet, data/val_test.parquet
```

---

## Training

Training uses RLVR with the CISPO algorithm via the bundled Tinker framework. The training script accepts CLI arguments using the `chz` configuration system.

The two reward-shaping signals of **ReViSQL-BIRD** — VeriEQL formal equivalence and evidence supervision — are controlled by the flags `use_verieql`, `use_evidence_supervision_prompt`, and `use_evidence_supervision_penalty` (all `True` by default). The base **ReViSQL** recipe sets all three to `False`, leaving only the result-based reward. Because VeriEQL is enabled by default, launch training with the `--extra verieql` group so the VeriEQL solver is installed.

**ReViSQL-BIRD-K2.6 (human-parity model — VeriEQL + evidence supervision):**
```bash
uv run --extra verieql scripts/train.py \
    model_name=moonshotai/Kimi-K2.6 \
    train_data_path=data/bird-verified-train.parquet \
    test_data_path=data/val_test.parquet \
    db_path=/path/to/bird/databases \
    loss_fn=cispo \
    group_size=16 \
    batch_size=64 \
    learning_rate=5e-5 \
    lora_rank=32 \
    max_turns=5 \
    max_output_tokens_per_turn=3072 \
    wandb_project=revisql \
    log_path=models/revisql-bird-k2.6
```

**ReViSQL-235B (base recipe — result-based reward only):**
```bash
uv run scripts/train.py \
    model_name=Qwen/Qwen3-235B-A22B \
    train_data_path=data/bird-verified-train.parquet \
    test_data_path=data/val_test.parquet \
    db_path=/path/to/bird/databases \
    loss_fn=cispo \
    group_size=16 \
    batch_size=64 \
    learning_rate=5e-5 \
    lora_rank=32 \
    max_turns=5 \
    max_output_tokens_per_turn=3072 \
    use_verieql=False \
    use_evidence_supervision_prompt=False \
    use_evidence_supervision_penalty=False \
    wandb_project=revisql \
    log_path=models/revisql-235b
```

Training configuration (matching the paper):

| Parameter | Value |
|---|---|
| LoRA rank | 32 |
| Batch size | 64 |
| Group size | 16 |
| Learning rate | 5 × 10⁻⁵ |
| Max turns | 5 |
| Max output tokens/turn | 3,072 |
| Train/val split | 85:15 |
| Algorithm | CISPO |
| VeriEQL reward (ReViSQL-BIRD) | enabled; soft 1 − β = 0.8 on non-equivalence (β = 0.2) |
| Evidence supervision (ReViSQL-BIRD) | enabled (prompt + penalty λ = 0.1) |
| Process reward (PRM) | none |

The training checkpoint with the highest validation accuracy on `data/val_test.parquet` is selected for inference.

---

## Inference

The inference script runs greedy decoding or temperature sampling to generate *N* SQL candidates per question; for *N* > 1 the final answer is selected by execution-based majority voting (self-consistency).

**Greedy decoding:**
```bash
uv run scripts/infer.py \
    model_name=moonshotai/Kimi-K2.6 \
    sampler_path=models/revisql-bird-k2.6/best_checkpoint \
    data_path=data \
    test_data=val_test.parquet \
    db_path=/path/to/bird/databases \
    output_path=outputs/revisql-bird-k2.6-greedy \
    run_name=revisql-bird-k2.6-greedy \
    temperature=0.0 \
    repeat=1
```

**Self-consistency (majority voting over 16 candidates):**
```bash
uv run scripts/infer.py \
    model_name=moonshotai/Kimi-K2.6 \
    sampler_path=models/revisql-bird-k2.6/best_checkpoint \
    data_path=data \
    test_data=val_test.parquet \
    db_path=/path/to/bird/databases \
    output_path=outputs/revisql-bird-k2.6-sc16 \
    run_name=revisql-bird-k2.6-sc16 \
    temperature=1.0 \
    repeat=16
```

Inference uses the same evidence-supervision system prompt as ReViSQL-BIRD training (`use_evidence_supervision_prompt=True` by default). VeriEQL and the evidence-supervision penalty are training-only reward-shaping signals and are automatically disabled on the test split.

---

## Reproducing Paper Results

### BIRD benchmarks (Arcwise-Plat-SQL, Arcwise-Plat-Full)

1. Fine-tune Kimi-K2.6 with ReViSQL-BIRD using the training command above.
2. Select the checkpoint with the highest validation accuracy on `data/val_test.parquet`.
3. Run inference on the `arcwise_plat_sql.json` and `arcwise_plat_full.json` test splits within `val_test.parquet`, under greedy decoding and self-consistency.

### Spider 2-SQLite

Run inference on `data/spider2sqlite_eval.parquet` with `sql_engine=SQLite`.

### Spider 2-Snow

Run inference on `data/spider_2_snow_eval.parquet` with `sql_engine=Snowflake` (requires Snowflake credentials in `.env`).

### Cross-benchmark generalization (ReViSQL-235B)

Fine-tune Qwen3-235B-A22B with the base ReViSQL recipe (result-based reward only) using the ReViSQL-235B command above, then run inference on the Arcwise-Plat and Spider 2 evaluation sets.
