# Variable-Length Semantic IDs for Recommender Systems

This is the official repository for the paper “Variable-Length Semantic IDs for Recommender Systems”.
It contains the full implementation of all proposed methods, along with the exact experimental configurations and detailed results required to reproduce the findings reported in the paper.

---

## 1. Environment setup

### 1.1 Install Python dependencies

```bash
pip install -r requirements.txt
```

Some optional acceleration libraries (`xformers`, `triton`) may require a compatible CUDA and compiler setup.

---

### 1.2 Hugging Face authentication (Amazon only)

Preprocessing the Amazon dataset requires downloading pretrained text models and computing EmbeddingGemma embeddings.
Please authenticate with Hugging Face:


```bash
huggingface-cli login
```

---

## 2. Dataset preparation

### 2.1 Downloading datasets

#### Yambda

We provide a helper for downloading user-item interactions and item embeddings:

```python
from scripts.data.yambda import download

download(dst_dir="./data/yambda")
```

This creates:

* `data/yambda/interactions.parquet`
* `data/yambda/embeddings.parquet`

---

#### VK-LSVD

Use the analogous helper function `download` from:

* `scripts/data/vklsvd.py`

---

#### Amazon (Amazon Reviews 2023)

Amazon preprocessing requires a manual download step:

1. Download **metadata** and **reviews** from
   [https://amazon-reviews-2023.github.io/](https://amazon-reviews-2023.github.io/)

2. Run the function `process` from the helper script:

* `scripts/data/amazon.py`

The Amazon helper also computes **Gemma embeddings** and therefore requires Hugging Face access.

---

### 2.2 Dataset preprocessing

Each dataset provides a preprocessing script that performs:

* data for training and evaluating both semantic ID construction methods and sequential recommendation models.

For **Yambda**:

```bash
python scripts/data/yambda.py
```

The output is written to the destination directory specified in `main()` (default: `./data/yambda`).

Additionally, the script writes a user subsample of the test interactions to (5\% for Yambda, 10\% for VK-LSVD) for RQ-2:

* `./data/yambda/seqrec_test_sample_interactions.parquet`

---

## 3. Running experiments

### 3.1 Training semantic ID models

We provide separate training scripts for each semantic-ID method:

```bash
python -m scripts.train_X --config path/to/config.yaml
```

where `X` is one of:

* `dvae`
* `reinforce`
* `rkmeans`

---

### 3.2 Training the sequential recommender

The sequential recommender is trained using:

```bash
python -m scripts.train_seqrec --config path/to/config.yaml
```

---

### 3.3 Configurations and outputs

* All configurations used for **RQ1–RQ4** are provided in `configs/`; for **RQ3**, we reuse dVAE results on Yambda from **RQ1**
* Evaluation outputs are stored under `results/`.

Each experiment can be reproduced by running the corresponding training script with the appropriate config file.

---

## 4. Reproducibility notes

* All experiments are fully configuration-driven via YAML files.
* Random seeds are fixed in preprocessing and training scripts.
* Evaluation follows the exact protocols described in the paper.
* We provide actual paper evaluation results under `results/` for reference.

---
