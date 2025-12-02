# 📘 Finetuning Models for SleepQA

This repository provides an updated, fully reproducible implementation of the **SleepQA** retriever–reader pipeline, based on the original code released by the authors. It includes:

- Modernized support for recent PyTorch & HuggingFace versions  
- Fixes for hybrid CPU/GPU training issues in the DPR framework  
- A new evaluation script for computing EM/F1 reader metrics  
- Clear documentation and step-by-step instructions  
- Explicit description of dependencies between steps  
- End-to-end commands to reproduce the results in the SleepQA paper  

The system is built on top of the **Facebook Dense Passage Retrieval (DPR)** framework and uses **Hydra** extensively for configuration.  
This README assumes no prior familiarity with DPR.

---

# 📑 Table of Contents

1. [Computational Requirements](#computational-requirements)  
2. [Repository Improvements & Modifications](#repository-improvements--modifications)  
3. [Overview of the Retriever→Reader Pipeline](#overview-of-the-retrieverreader-pipeline)  
4. [Hydra Configuration Essentials](#hydra-configuration-essentials)  
5. [Dependency Flow Between Steps](#dependency-flow-between-steps)  
6. [Steps to Reproduce the SleepQA Paper](#steps-to-reproduce-the-paper)  
   - Step 1 — Environment Setup  
   - Step 2 — Train Retriever  
   - Step 3 — Build Corpus Embeddings  
   - Step 4 — Validate Retriever  
   - Step 5 — Build Reader Training Data  
   - Step 6 — Train Reader  
   - Step 7 — Validate Reader  
   - Step 8 — Evaluate Reader EM/F1  
7. [Reproducing the Full Retriever→Reader Pipeline](#replicating-the-pipeline)  
8. [Citation](#citation)

---

# ⚙️ Computational Requirements

Experiments were replicated using AWS infrastructure.

## Recommended Hardware

| Component | Specification |
|----------|---------------|
| Instance | `ml.g6.xlarge` |
| Storage | 500 GB |
| CPU | 4 vCPUs (AMD EPYC 7R13) |
| System RAM | 16 GB |
| GPU | NVIDIA L4 Tensor Core |
| GPU Memory | 24 GB |

**Estimated runtime:**  
A full retriever + reader fine-tuning run requires **8–10 hours** on this configuration.

---

# 🛠 Repository Improvements & Modifications

Several changes were required to support modern libraries and SleepQA-specific tasks.

### ✔ New Script: `utils/evaluate_reader_predictions.py`
Added to compute:

- Exact Match (EM)
- F1 Score
- Reader recall@K

The original DPR code did *not* produce these metrics.

---

### ✔ Modernization fixes

#### `dpr/models/hf_models.py`
- Updated deprecated HuggingFace tokenizer args  
- Updated AdamW import for latest PyTorch  

#### `dpr/utils/model_utils.py`
- Added support for loading “weights-only” checkpoints  
- Modern PyTorch requires `weights_only=False`

#### `train_dense_encoder.py`
- Fixed runtime crashes in hybrid CPU/GPU setups  
- Ensured all model parameters migrate to GPU consistently

---

# 🔄 Overview of the Retriever→Reader Pipeline

The SleepQA pipeline follows the standard DPR architecture:

```
Question → [Retriever Bi-Encoder] → Top-K Passages → [Extractive Reader] → Answer Span
```

## Retriever (Bi-Encoder)
- Embeds questions & passages  
- Retrieves top-K most relevant documents  
- Evaluated using **Recall@K**

## Reader (Extractive QA)
- Given the retrieved passages, extracts the exact answer span  
- Evaluated using **Exact Match (EM)** and **F1**

The entire pipeline is trained and evaluated in sequence.

---

# 🧰 Hydra Configuration Essentials

DPR uses Hydra extensively; only a few parameters need to change.

### `encoder`
Chooses the pretrained model to fine-tune:

- `hf_SciBERT`
- `hf_PubMedBERT`
- `hf_ClinicalBERT`
- `hf_biobert`
- `hf_BioASQ`
- `hf_bert-base-uncased`

Defaults are stored in `DPR-main/conf/encoder/`.

---

### `ctx_source`
Document corpus for retrieval.

Use:
- `dpr_sleep`

---

### `train_datasets` / `dev_datasets`
Defined in `DPR-main/conf/datasets`:

- `sleep-train`
- `sleep-dev`
- `sleep-test`

---

# 🔗 Dependency Flow Between Steps

The entire reproduction pipeline is strictly sequential.  
Each step relies on outputs from earlier steps.

Below is the *true dependency graph*:

```
Step 2 → Best Retriever Checkpoint
      ↓
Step 3 → Corpus Embeddings
      ↓
Step 4 → Retriever Recall Metrics

Step 2 + Step 3
      ↓
Step 5 → Reader Train/Dev JSONs
      ↓
Step 6 → Best Reader Checkpoint
      ↓
Step 7 → Reader Predictions JSON
      ↓
Step 8 → EM/F1 Metrics

Final Pipeline Reproduction:
  Uses Step 2 (best retriever) + Step 6 (best reader)
```

This dependency structure is explicitly captured in each step below.

---

# 🚀 Steps to Reproduce the Paper

Each step includes:

- What the step does  
- Why it matters  
- Inputs from previous steps  
- Outputs used in later steps

---

# Step 1 — Setting Up Environment and Dependencies

### What it does
Installs DPR, Hydra configs, HF models, and spaCy model.

### Why it matters
Missing dependencies cause tokenization and data pipeline failures.

### Command
```bash
python setup.py install
python -m spacy download en_core_web_sm
```

---

# Step 2 — Train the Retriever  
**(First major model in the pipeline)**

### What it does
Fine-tunes the bi-encoder retriever using SleepQA train/dev datasets.

### Inputs
- `sleep_train` and `sleep_dev` datasets  
- Selected encoder (e.g., `hf_biobert`)

### Outputs (needed for Steps 3, 4, 5)
- **Best retriever checkpoint**, e.g.  
  `dpr_bioencoder.29`

### Command
```bash
python train_dense_encoder.py \
    encoder=hf_biobert \
    train_datasets=[sleep_train] \
    dev_datasets=[sleep_dev] \
    train.num_train_epochs=30 \
    train.batch_size=16 \
    train.hard_negatives=0 \
    train.other_negatives=1 \
    output_dir=<OUTPUT_DIR>
```

---

# Step 3 — Generate Corpus Embeddings  
**(Uses retriever from Step 2)**

### What it does
Encodes every document in the SleepQA corpus into embeddings.

### Inputs
- Best retriever checkpoint (from Step 2)  
- `dpr_sleep` corpus

### Outputs (needed for Steps 4 and 5)
- **Corpus embedding files**, e.g.:  
  `/encoder/corpus-encoding/embeds_*`

### Command
```bash
python generate_dense_embeddings.py \
    encoder=hf_biobert \
    model_file=/home/user/bio-bert/encoder/dpr_bioencoder.29 \
    out_file=<PATH_TO_EMBEDS> \
    ctx_src=dpr_sleep
```

---

# Step 4 — Validate Retriever (Recall@K)  
**(Uses Steps 2 + 3)**

### What it does
Measures how often the retriever returns the gold document in top-K.

### Inputs
- Retriever checkpoint (Step 2)  
- Corpus embeddings (Step 3)  
- `sleep_test` dataset

### Outputs
- Recall@1 (or Recall@K) metrics  
*(does not feed into later steps)*

### Command
```bash
python dense_retriever.py \
    encoder=hf_biobert \
    model_file=<BEST_RETRIEVER> \
    qa_dataset=sleep_test \
    ctx_datatsets=[dpr_sleep] \
    encoded_ctx_files=["<EMBEDS>*"] \
    out_file=<RETRIEVER_OUTPUT_FILE>
```

---

# Step 5 — Generate Reader Training Data  
**(Uses Steps 2 + 3)**

### What it does
Uses the retriever to collect top-K candidate passages for each training question, forming reader supervision data.

### Inputs
- Best retriever checkpoint (Step 2)  
- Corpus embeddings (Step 3)

### Outputs (used in Step 6)
- `sleep_train.json`  
- `sleep_dev.json`

### Command (train example)
```bash
python dense_retriever.py \
    encoder=hf_biobert \
    qa_dataset=sleep_train \
    out_file=<TRAIN_JSON> \
    ctx_datatsets=['dpr_sleep'] \
    encoded_ctx_files='[<EMBEDS>*]' \
    model_file=<BEST_RETRIEVER>
```

---

# Step 6 — Train the Reader  
**(Uses Step 5)**

### What it does
Fine-tunes the span-extraction reader using retrieved passages.

### Inputs
- `sleep_train.json` (from Step 5)  
- `sleep_dev.json` (from Step 5)  

### Outputs (used in Steps 7 & Pipeline reproduction)
- **Best reader checkpoint**, e.g.  
  `dpr_extractive_reader.6.16`

### Command
```bash
python train_extractive_reader.py \
    encoder=hf_biobert \
    train_files=<TRAIN_JSON> \
    dev_files=<DEV_JSON> \
    output_dir=<READER_OUTPUT_DIR>
```

---

# Step 7 — Validate Reader on Gold Oracle Test Set  
**(Uses Step 6)**

### What it does
Generates predicted answer spans for the oracle test set.

### Inputs
- Best reader checkpoint (Step 6)  
- Oracle test dataset  
- `train_files=null` (inference mode)

### Outputs (used in Step 8)
- `reader_predictions.json`

### Command
```bash
python train_extractive_reader.py \
    encoder=hf_biobert \
    model_file=<BEST_READER> \
    dev_files=<ORACLE_SLEEP_TEST> \
    train_files=null \
    prediction_results_file=<READER_PRED_OUTPUT>
```

---

# Step 8 — Evaluate Reader EM/F1  
**(Uses Step 7)**

### What it does
Computes the official **EM** and **F1** metrics for the reader.

### Inputs
- `reader_predictions.json` (from Step 7)

### Outputs
- Final EM  
- Final F1  
- Reader recall@K (optional)

### Command
```bash
python evaluate_reader_predictions.py <READER_PRED_OUTPUT>
```

---

# 🧩 Replicating the Full Retriever→Reader Pipeline

To reproduce **Pipeline-1 EM/F1** (as in the SleepQA paper):

### 1. Convert checkpoints to PyTorch format  
Uses outputs from **Step 2** and **Step 6**.

### 2. Update paths in `qa_system.py`

### 3. Run the pipeline end-to-end  
This yields the final EM/F1 reported in the paper.

Commands:

Retriever encoders:
```bash
python convert_dpr_original_checkpoint_to_pytorch.py \
    --type ctx_encoder \
    --src <BEST_RETRIEVER> \
    --dest <DEST_FOLDER>

python convert_dpr_original_checkpoint_to_pytorch.py \
    --type question_encoder \
    --src <BEST_RETRIEVER> \
    --dest <DEST_FOLDER>
```

Reader:
```bash
python convert_dpr_original_checkpoint_to_pytorch.py \
    --type reader \
    --src <BEST_READER> \
    --dest <DEST_FOLDER>
```

Final run:
```bash
python qa_system.py
```

---

# 📚 Citation

If you use this repository or reproduce the SleepQA results, please cite:

**Bojic et al. (2022).**  
*SleepQA: Clinical Question Answering for Sleep Medicine Using Domain-Specific Pretrained Language Models.*  
Proceedings of Machine Learning Research.

