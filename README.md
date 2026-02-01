# Image Safety Scanner

A Python command-line tool that scans folders of images for unsafe content using Google's [ShieldGemma 2](https://huggingface.co/google/shieldgemma-2-4b-it) multimodal AI model. Each image is classified across three safety dimensions — **sexually explicit**, **dangerous**, and **violence/gore** — and the results are written to a human-readable report file.

Built for use cases like auditing game asset libraries, moderating user-uploaded images, or pre-screening visual content before publication.

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Output Format](#output-format)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Development](#development)
- [Model Attribution](#model-attribution)
- [License](#license)

---

## Features

- **Three-category safety classification** — detects sexually explicit material, dangerous content, and violence/gore
- **Local inference** — all processing runs on your machine; no images are uploaded to external services
- **GPU acceleration** — automatically uses CUDA when available, falls back to CPU
- **Batch processing** — scans an entire folder of images in one run (up to a configurable limit)
- **Structured reports** — outputs per-image predictions and raw probability scores to a text file
- **Error tolerance** — gracefully skips non-image files without crashing

---

## How It Works

```
Image folder          ShieldGemma 2 model          Report file
 ┌──────────┐        ┌───────────────────┐        ┌──────────────────────┐
 │ img1.png │───────▶│  Preprocess with  │───────▶│ File name: img1.png  │
 │ img2.jpg │        │  AutoProcessor    │        │ Prediction: Safe     │
 │ img3.png │        │        │          │        │ Probabilities: ...   │
 │  ...     │        │        ▼          │        │                      │
 └──────────┘        │  Run inference    │        │ File name: img2.jpg  │
                     │  (GPU or CPU)     │        │ Prediction: May      │
                     │        │          │        │   contain: gory      │
                     │        ▼          │        │   material           │
                     │  Apply 0.5        │        │ Probabilities: ...   │
                     │  threshold per    │        │  ...                 │
                     │  category         │        └──────────────────────┘
                     └───────────────────┘
```

1. **`folder_walker()`** iterates over every file in the target directory.
2. **`image_classifire()`** loads each image with PIL, preprocesses it using the model's `AutoProcessor`, and runs inference with `ShieldGemma2ForImageClassification`.
3. **`threshold()`** converts the raw probability scores into a human-readable verdict using a **0.5 probability threshold** per category.
4. Results (filename, prediction, raw probabilities) are written to `results/<folder_name>_result.txt`.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.8 or higher |
| **Hugging Face account** | Required to download the ShieldGemma 2 model weights |
| **Gemma Terms of Use** | Must be accepted at the [model page](https://huggingface.co/google/shieldgemma-2-4b-it) before first use |
| **Hardware** | CUDA-capable GPU recommended; CPU works but is significantly slower |
| **Disk space** | ~8 GB for the 4B-parameter model weights (downloaded once) |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/tlee817/image-safety-scanner.git
cd image-safety-scanner
```

### 2. Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install torch transformers pillow tqdm
```

> **Note (GPU):** If you have an NVIDIA GPU and want CUDA acceleration, install the CUDA-enabled version of PyTorch by following the instructions at [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### 4. Authenticate with Hugging Face

```bash
pip install huggingface_hub
huggingface-cli login
```

You will be prompted for an access token. Generate one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Make sure you have first accepted the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) on the model page.

---

## Usage

### Quick start

```bash
python image_process.py /path/to/your/image/folder
```

For example, to scan the included sample images:

```bash
python image_process.py data/Game1
python image_process.py data/Gladihoppers
```

The first run will download the ShieldGemma 2 model weights (~8 GB). Subsequent runs use the cached model.

### What happens

- The script scans every file in the specified folder (non-image files are skipped).
- A progress log is printed to the console (`File found: ...`).
- A report file named `<folder_name>_result.txt` is created in the `results/` directory.

---

## Output Format

The report file contains one block per image:

```
File name: example.png
Prediction: Safe
Probabilities: tensor([[0.9964, 0.0036],
        [0.9928, 0.0072],
        [0.9931, 0.0069]])
```

### Prediction values

| Prediction | Meaning |
|---|---|
| `Safe` | No category exceeded the 0.5 threshold |
| `May contain: pornographic material` | Sexually explicit score > 0.5 |
| `May contain: dangerous material` | Dangerous content score > 0.5 |
| `May contain: gory material` | Violence/gore score > 0.5 |
| `May contain: pornographic, dangerous, gory material` | Multiple categories flagged |

### Probability tensor

Each row in the probability tensor corresponds to one safety category (in order):

| Row | Category | Column 0 (Safe) | Column 1 (Unsafe) |
|---|---|---|---|
| 0 | Sexually explicit | P(safe) | P(unsafe) |
| 1 | Dangerous content | P(safe) | P(unsafe) |
| 2 | Violence / gore | P(safe) | P(unsafe) |

A category is flagged when its **Column 1 (Unsafe)** value exceeds **0.5**.

---

## Configuration

Configuration is done by editing constants at the top of `image_process.py`:

| Constant | Default | Description |
|---|---|---|
| `MAX_IMAGES` | `100` | Maximum number of images to process per folder |
| `model_id` | `"google/shieldgemma-2-4b-it"` | Hugging Face model identifier |

The classification threshold is hardcoded at **0.5** inside the `threshold()` function.

---

## Project Structure

```
image-safety-scanner/
├── image_process.py            # Main script — model loading, scanning, classification
├── LICENSE                     # MIT License
├── README.md                   # This file
├── .gitignore                  # Excludes venvs, caches, secrets, IDE files
├── .pre-commit-config.yaml     # Pre-commit hooks (detect-secrets, whitespace, large files)
│
├── data/                       # Reference data and sample images
│   ├── policy_rows_seen.json   # Probability tensor order and sample values
│   ├── Game1/                  # Sample image folder (133 images — game assets)
│   └── Gladihoppers/           # Sample image folder (22 images — game sprites)
│
└── results/                    # Scan output reports
    ├── .gitkeep                # Keeps directory in git when empty
    ├── Game1_result.txt        # Sample output from scanning data/Game1/
    └── Gladihoppers_result.txt # Sample output from scanning data/Gladihoppers/
```

---

## Development

### Pre-commit hooks

This project uses [pre-commit](https://pre-commit.com/) for code hygiene:

```bash
pip install pre-commit
pre-commit install
```

Configured hooks:

| Hook | Purpose |
|---|---|
| **detect-secrets** (v1.6.0) | Prevents accidental commit of API keys, tokens, or passwords |
| **trailing-whitespace** | Removes trailing whitespace from files |
| **end-of-file-fixer** | Ensures files end with a newline |
| **check-added-large-files** | Blocks commits of unexpectedly large files |

---

## Model Attribution

This project uses the [ShieldGemma 2-4B-IT](https://huggingface.co/google/shieldgemma-2-4b-it) model by Google, released under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Model weights are **not** redistributed in this repository; users must accept the Gemma Terms of Use to download and run the model.

```bibtex
@misc{zeng2025shieldgemma2robusttractable,
    title={ShieldGemma 2: Robust and Tractable Image Content Moderation},
    author={Wenjun Zeng and Dana Kurniawan and Ryan Mullins and Yuchi Liu
            and Tamoghna Saha and Dirichi Ike-Njoku and Jindong Gu
            and Yiwen Song and Cai Xu and Jingjing Zhou and Aparna Joshi
            and Shravan Dheep and Mani Malek and Hamid Palangi
            and Joon Baek and Rick Pereira and Karthik Narasimhan},
    year={2025},
    eprint={2504.01081},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2504.01081},
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2025 tlee817
