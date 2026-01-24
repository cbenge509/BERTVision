# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BERTVision is a research project presenting a parameter-efficient approach for Question Answering (QA) and text classification using BERT hidden state activations. The key innovation is leveraging internal transformer layer outputs (typically discarded during inference) to enable efficient fine-tuning with reduced computational requirements.

## GitHub Pages Website

The project website is deployed at https://cbenge509.github.io/BERTVision/

**Branch structure:**
- `master` - Research code (Python/PyTorch/TensorFlow models)
- `gh-pages` - Astro website for project documentation

**Tech stack (`gh-pages` branch):**
- Astro 5.x (static site generator)
- Tailwind CSS v4 via `@tailwindcss/vite` plugin
- TypeScript (strict mode)
- Vitest (unit tests)
- Playwright (E2E tests)
- gts (Google TypeScript Style)

**Website structure:**
```
src/
├── components/     # Astro components (Nav, Hero, CodeBlock, etc.)
├── layouts/        # Base layout template
├── pages/          # Page routes (index.astro)
├── styles/         # Global CSS and theme
├── content/        # Results data (JSON)
├── lib/            # Utility functions
└── assets/images/  # Optimized images
public/
└── paper/          # Research paper PDF
tests/
└── e2e/            # Playwright E2E tests
```

**Development commands:**
```bash
npm install          # Install dependencies
npm run dev          # Start dev server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run linter
npm run test         # Run unit tests
npm run test:e2e     # Run E2E tests
```

Changes pushed to `gh-pages` automatically deploy via GitHub Actions.

## Build & Environment Setup

**Package Manager**: Pipenv (Python 3.7)

**Recommended Conda setup** (from `code/torch/README.md`):
```bash
conda create -n my_ml python=3.7 pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
conda activate my_ml
conda install -c conda-forge jupyter numpy pandas matplotlib scikit-learn pip transformers datasets tqdm
pip install loguru hyperopt
```

**Alternative with Pipenv**:
```bash
pipenv install
pipenv shell
```

## Running Models

All PyTorch models are executed as Python modules from `code/torch/`:

```bash
cd code/torch

# BERT fine-tuning for GLUE tasks
python -m models.bert_glue --model MSR --checkpoint bert-base-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 32

# AdapterPooler (parameter-efficient) for GLUE
python -m models.ap_glue --model AP_MSR --checkpoint bert-base-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 32

# SQuAD 2.0 QA
python -m models.bert_squad --model SQuAD --checkpoint bert-large-uncased --lr 2e-5 --max-seq-length 384 --batch-size 8
python -m models.ap_squad --model AP_SQuAD --checkpoint bert-large-uncased --lr 2e-5 --max-seq-length 384 --batch-size 8

# Hyperparameter search
python -m models.hypersearch --model MSR --checkpoint bert-large-uncased --batch-size 32 --num-labels 2 --max-seq-length 128
python -m models.ap_hypersearch --model AP_STSB --checkpoint bert-base-uncased --batch-size 16 --num-labels 1 --max-seq-length 128

# Parameter freezing experiments
python -m models.pfreezing --model MSR --checkpoint bert-base-uncased --batch-size 32 --lr 2e-5 --num-labels 2 --max-seq-length 128
```

**Error analysis mode** (outputs predicted vs true labels):
```bash
python -m models.bert_glue --model MSR --checkpoint bert-base-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 32 --error
```

## Generating Embeddings

Run from `code/torch/gen_embeds/`:

```bash
cd code/torch/gen_embeds

# GLUE task embeddings
python qqpairs_embeds.py --checkpoint bert-base-uncased --lr 1e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
python mnli_embeds.py --checkpoint bert-large-uncased --lr 1e-5 --num-labels 3 --max-seq-length 128 --batch-size 32

# SQuAD embeddings
python squad_embeds.py --checkpoint bert-large-uncased --lr 2e-5 --max-seq-length 384 --batch-size 8
```

## Running Tests

Tests use Python's unittest framework:
```bash
cd code/tensorflow
python -m unittest discover
# or single test file:
python tests/test_image_generator.py
```

## Architecture

### Dual Implementation
- **PyTorch** (`code/torch/`) - Primary implementation, more feature-complete
- **TensorFlow** (`code/tensorflow/`) - Alternative implementation

### PyTorch Code Structure (`code/torch/`)

```
models/           # Entry points (run with python -m models.MODEL_NAME)
  bert_glue/      # BERT fine-tuning for GLUE benchmark
  bert_squad/     # BERT for SQuAD 2.0
  ap_glue/        # AdapterPooler for GLUE
  ap_squad/       # AdapterPooler for SQuAD
  hypersearch/    # Hyperparameter tuning (BERT)
  ap_hypersearch/ # Hyperparameter tuning (AP)
  pfreezing/      # Parameter freezing experiments
  parasite/       # Parasite learning mechanism

common/
  trainers/       # Training loops for each model variant
  evaluators/     # Evaluation logic for different tasks

data/
  bert_processors/  # Raw text → tokenized BERT input (GLUE tasks)
  h5_processors/    # HDF5 embeddings → AdapterPooler input

utils/
  bert_models.py    # BERT architecture variants
  compress_utils.py # AdapterPooler and compression utilities
  collate.py        # Data collation functions
  squad_preprocess.py

gen_embeds/       # Scripts to generate embeddings for each dataset
```

### Data Flow

1. **BERT embeddings extraction**: Fine-tune BERT, capture hidden state activations from all transformer layers
2. **Storage**: Embeddings stored as HDF5 files (386 tokens × 1024 features for BERT-large)
3. **AdapterPooler training**: Train lightweight adapter on pre-extracted embeddings

### Supported Tasks

**GLUE Benchmark**: MSR, CoLA, MNLI, QNLI, QQP, RTE, SST, STSB, WNLI
**QA**: SQuAD 2.0

### Model-specific Hyperparameters

| Task | num-labels | lr | batch-size | max-seq-length |
|------|------------|-----|------------|----------------|
| MNLI | 3 | 1e-5 | 32 | 128 |
| QQP/QNLI | 2 | 1e-5 | 32 | 128 |
| RTE | 2 | 2e-5 | 16 | 250 |
| SST/MSR/CoLA | 2 | 2e-5 | varies | 128 |
| STSB | 1 | 2e-5 | 16 | 128 |
| SQuAD | - | 2e-5 | 8-16 | 384 |

### Key Dependencies
- `torch==1.7.1`, `tensorflow==2.2.0`
- `transformers==2.10.0` (Hugging Face)
- `h5py==2.10.0` (embedding storage)
- `hyperopt` (hyperparameter search)
- `loguru` (logging)
