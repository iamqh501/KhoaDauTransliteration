# Applying NLP Techniques to Khoa Dau Transliteration

A bidirectional Natural Language Processing (NLP) system for transliterating between modern Vietnamese **Quốc Ngữ** (Latin script) and **Khoa Đẩu** script (encoded in the Unicode Private Use Area `U+E000`–`U+E026`).

The project investigates and benchmarks deterministic, dictionary-vectorized, and deep learning paradigms across two fundamentally asymmetric linguistic directions:
1. **Quốc Ngữ $\rightarrow$ Khoa Đẩu (`quoc_ngu_to_khoa_dau`)**: A deterministic, high-locality phonological transformation solved with **100.00% accuracy** via Rule-Based phonological algorithms, CPU/GPU Dictionary Vectorization (~530,000 words/sec), and an ultra-compact **1D Convolutional Neural Network (`KhoaDauCNN`)** requiring only **3,326 parameters (16.70 KB)**.
2. **Khoa Đẩu $\rightarrow$ Quốc Ngữ (`khoa_dau_to_quoc_ngu`)**: An ambiguous one-to-many contextual reconstruction and tone-restoration problem solved with a **Character-Level Seq2Seq Transformer** (`12.86M` parameters) trained on **~2.27 million** deduplicated Vietnamese sentence pairs, achieving a **0.65% Character Error Rate (CER)** and **2.18% Word Error Rate (WER)** with GPU/CPU-vectorized Beam Search ($k=3$).

---

## Table of Contents
- [1. Linguistic & Orthographic Background](#1-linguistic--orthographic-background)
- [2. System Architecture & Methodology](#2-system-architecture--methodology)
  - [2.1 Module 1: Quốc Ngữ to Khoa Đẩu](#21-module-1-quốc-ngữ-to-khoa-đẩu)
  - [2.2 Module 2: Khoa Đẩu to Quốc Ngữ](#22-module-2-khoa-đẩu-to-quốc-ngữ)
  - [2.3 Data Engineering Pipeline](#23-data-engineering-pipeline)
- [3. Experimental Results & Benchmarks](#3-experimental-results--benchmarks)
  - [3.1 Quốc Ngữ to Khoa Đẩu Benchmark](#31-quốc-ngữ-to-khoa-đẩu-benchmark)
  - [3.2 Khoa Đẩu to Quốc Ngữ Evaluation & Phased Progression](#32-khoa-đẩu-to-quốc-ngữ-evaluation--phased-progression)
  - [3.3 Beam Search Decoding Trade-off Analysis](#33-beam-search-decoding-trade-off-analysis)
- [4. Interactive Desktop Applications (GUI)](#4-interactive-desktop-applications-gui)
- [5. Repository Structure](#5-repository-structure)
- [6. Installation & Usage (Windows Command Prompt)](#6-installation--usage-windows-command-prompt)

---

## 1. Linguistic & Orthographic Background

The Khoa Đẩu script encoded in this project represents Vietnamese syllables using **36 Private Use Area (PUA) glyphs** (`U+E000`–`U+E026`, excluding unassigned slots `U+E00D`, `U+E010`, and `U+E023`), rendered via the bundled OpenType font `misc/font/KhoaDau-Regular_v3.otf`.

Converting Vietnamese Quốc Ngữ into Khoa Đẩu involves five phonological operations:
1. **Tone Stripping**: Tone marks (*huyền, sắc, hỏi, ngã, nặng*) are omitted while base vowel qualities (`ă, â, ê, ô, ơ, ư`) and `đ` are retained.
2. **Final-Coda Locking**: Syllable-final consonants and semivowels map to dedicated terminal glyphs distinct from their initial/medial counterparts:
   - `ng / nh` $\rightarrow$ `U+E015`, `o / u` $\rightarrow$ `U+E00F`, `i / y` $\rightarrow$ `U+E00E`, `m` $\rightarrow$ `U+E024`, `n` $\rightarrow$ `U+E025`, `t` $\rightarrow$ `U+E026`.
3. **Virtual Onset Insertion**: Zero-onset syllables (not starting with a consonant or locked `o/u`) receive a leading virtual onset glyph `U+E006`.
4. **Pre-Onset Vowel Reordering**: Specific nuclei (`ươ, ô, ê, ơ, e, â`) shift before the onset consonant unless locked inside protected vowel clusters (`iê, yê, ia, uô, uơ`).
5. **Phonemic Glyph Compression (Homophone Merging)**: Multiple Quốc Ngữ graphemes collapse into shared phonemic glyphs:
   - `c / k / q` $\rightarrow$ `U+E012`
   - `d / gi / r` $\rightarrow$ `U+E013`
   - `ch / tr` $\rightarrow$ `U+E01D`
   - `ng / ngh` $\rightarrow$ `U+E01F`
   - `g / gh` $\rightarrow$ `U+E015`
   - `b / p` $\rightarrow$ `U+E011`
   - `s / x` $\rightarrow$ `U+E01C`
   - `i / y` $\rightarrow$ `U+E005`
   - `â / ơ` $\rightarrow$ `U+E002`
   - `uô / uơ / uâ` $\rightarrow$ `U+E00C`

Because tone marks are stripped and multiple consonantal/vocalic graphemes collapse into shared glyphs, **forward transliteration (Quốc Ngữ $\rightarrow$ Khoa Đẩu) is deterministic**, whereas **reverse transliteration (Khoa Đẩu $\rightarrow$ Quốc Ngữ) requires sentence-level semantic and syntactic context** to disambiguate homophones and reconstruct diacritics.

---

## 2. System Architecture & Methodology

### 2.1 Module 1: Quốc Ngữ to Khoa Đẩu

Three implementation strategies are provided and benchmarked:
- **Rule-Based Engine (`rule_based.py`)**: Exact phonological finite-state pipeline combined with **Dictionary Filtering** (`encode_safe`) over **17,974 normalized Vietnamese syllables** (`all-vietnamese-syllables.txt`). Non-Vietnamese tokens (English terms, proper nouns, numbers, punctuation) bypass phonetic conversion via $O(1)$ hash-set filtering, preventing foreign-word corruption in mixed-language text.
- **Vectorized Dictionary Lookup (`evaluate.py` / `parallel_utils.py`)**: Pre-computes syllable mappings (`all-vietnamese-syllables-encoded.csv`) for $O(1)$ CPU dictionary lookup or parallel GPU tensor indexing (`gpu_targets[safe_ids]`).
- **1D Convolutional Neural Network (`KhoaDauCNN` in `models.py`)**:
  - **Architectural Rationale**: Unlike RNNs (sequential bottleneck) or Transformers (quadratic attention overhead unnecessary for intra-syllable context), a stacked 1D CNN operates as a parallel sliding window over fixed-length syllables (`MAX_LEN = 7`).
  - **Design Findings**: Because closed-set syllable mapping is deterministic, **disabling Dropout (`dropout = 0.0`)** accelerates convergence and eliminates stochastic residual errors. A **2-layer, kernel-size-5** topology (`shallow_big`, `embed_dim = 8`, `hidden_dim = 16`, receptive field $= 9 > 7$) outperforms deeper 3-layer kernel-3 variants, achieving **100.00% exact syllable accuracy** across the entire Vietnamese syllable lexicon in just **16.70 KB**.

### 2.2 Module 2: Khoa Đẩu to Quốc Ngữ

Reverse transliteration is formulated as **Character-Level Neural Machine Translation** using an Encoder-Decoder **Seq2Seq Transformer** (`khoa_dau_to_quoc_ngu/src/models.py`):
- **Character-Level Tokenizer (`CharTokenizer`)**: Operates over a compact vocabulary of **152 characters** (`<pad>`, `<sos>`, `<eos>`, `<unk>`, 36 Khoa Đẩu PUA glyphs, Vietnamese accented Latin letters, digits, and punctuation). Character-level granularity eliminates Out-of-Vocabulary (OOV) failures and enables natural **Identity Copying** of interleaved foreign Latin words and numbers.
- **Seq2Seq Transformer Backbone**:
  - Phase 3 Production Configuration (`best_model_phase3.pth`): `D_MODEL = 512`, `N_HEAD = 16`, `NUM_ENCODER_LAYERS = 3`, `NUM_DECODER_LAYERS = 3`, `DIM_FEEDFORWARD = 512`, `GELU` activations, `MAX_LEN = 192` (**12,858,520 parameters**).
  - Phase 2 Compact Configuration (`best_model.pth`): `D_MODEL = 256`, `N_HEAD = 8`, `3` encoder/decoder layers (**4,071,576 parameters**, tuned for 4 GB VRAM hardware).
- **Training Throughput & Stability Optimizations**:
  - **In-RAM Pre-Tokenization**: Encodes all ~2.04M training sentence pairs into Python integer lists (`list[int]`) at dataset initialization, eliminating per-batch CPU string-processing bottlenecks and avoiding PyTorch shared-memory file-descriptor leaks.
  - **Dynamic Per-Batch Padding (`collate_fn`)**: Pads each mini-batch only to the maximum sequence length within that batch rather than static `MAX_LEN = 192`, cutting A100 VRAM consumption from ~33 GB to ~20 GB and increasing training throughput from **1.13 it/s to ~7.1 it/s**.
  - **FP16 Automatic Mixed Precision (AMP) with FP32 Loss Guard**: Runs forward passes under `torch.amp.autocast(dtype=torch.float16)` while casting output logits back to `float32` prior to `CrossEntropyLoss` to prevent log-softmax overflow (`NaN`/`Inf`), paired with `GradScaler` and gradient norm clipping (`1.0`).
  - **SDPA / Flash Attention Safety**: Uses `src_mask = None` with uniform `torch.bool` causal and padding masks and strict `MAX_LEN` sequence bounds to prevent CUDA illegal memory access faults.
- **Vectorized Beam Search Decoder (`beam_search_decode`)**: Batches all $k$ active candidate beams into a single tensor (`[curr_len, num_active]`) with expanded encoder memory (`[src_len, num_active, d_model]`) for a single parallel decoder pass per timestep, scored with length normalization $\text{score} / |\text{seq}|^{0.7}$.

### 2.3 Data Engineering Pipeline

Constructed from the `BlossomsAI/vietnamese-corpus` (~4 million raw sentences, ~24 GB):
1. **Sharded Parquet Extraction (`download_and_extract_hf.py`)**: Streams Hugging Face records into resumable 1M-row local Parquet shards.
2. **Orthographic Normalization & Cleaning (`data_maker.py`)**:
   - Strips non-Latin scripts (CJK, Greek, Cyrillic) via a strict Vietnamese-Latin-digit-punctuation whitelist to prevent vocabulary bloat.
   - Standardizes all Vietnamese syllables to **Modern Vietnamese Tone Placement** (`oà, oè, uý` via `normalize_tone_to_modern`) and Unicode `NFC`, matching the normalized `all-vietnamese-syllables.txt` lexicon.
   - Applies **Punctuation Padding** (`"đi học."` $\rightarrow$ `"đi học ."`), ensuring clean whitespace word boundaries for dictionary lookup and providing a consistent syllable-boundary stop signal for Transformer self-attention.
   - Encodes valid Vietnamese syllables into Khoa Đẩu while keeping foreign Latin tokens and numbers unchanged (teaching the model identity copying).
3. **Deduplication & Splitting (`prepare_final_dataset.py`)**: Drops duplicates and short/empty fragments to prevent memorization bias, yielding **~2.27 million unique sentence pairs** split **90% Train (~2.04M) / 5% Validation (~113K) / 5% Test (~113K)**.

---

## 3. Experimental Results & Benchmarks

### 3.1 Quốc Ngữ to Khoa Đẩu Benchmark

Evaluated on the `quoc_ngu_to_khoa_dau/data/source.txt` benchmark corpus (**204,758 words / 213,118 tokens**) and verified for exact syllable accuracy against all **17,974** standard Vietnamese syllables:

| Method | Syllable Accuracy | Benchmark Latency | Parameters | Disk Footprint | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dictionary (GPU Tensor Lookup)** | **100.00%** | **~399 ms** | N/A | 315.26 KB | Highest throughput (~530k tokens/sec) |
| **Dictionary (CPU Serial)** | **100.00%** | ~413 ms | N/A | 315.26 KB | Optimal CPU lookup without IPC overhead |
| **`KhoaDauCNN` — `Shallow_Big`** | **100.00%** | **~885 ms** | **3,326** | **16.70 KB** | **Champion AI model** (2 layers, kernel 5) |
| **`KhoaDauCNN` — `Big`** | 99.93% | ~1,253 ms | 3,342 | 17.23 KB | 3 layers, kernel 3, dims 8/16 (13/17,974 errors) |
| **Rule-Based (`encode_safe` Serial)** | **100.00%** | ~1,295 ms | N/A | 7.11 KB | Reference phonological implementation |
| **`KhoaDauCNN` — `Shallow_Small`** | 75.28% | < 850 ms | 1,202 | 8.37 KB | Extreme compression ablation (dims 4/8) |
| **`KhoaDauCNN` — `Small`** | 66.73% | < 1,200 ms | 1,210 | 8.81 KB | Extreme compression ablation (dims 4/8) |

> **Platform Note on CPU Multiprocessing**: On Windows (`spawn` process model), `ProcessPoolExecutor` incurs ~6,600 ms process creation and IPC serialization overhead on this workload; single-process dictionary lookup or batched CNN inference (`batch_size = 4096`) is significantly faster.

### 3.2 Khoa Đẩu to Quốc Ngữ Evaluation & Phased Progression

Development followed a strict gate-driven 4-phase engineering protocol evaluated on held-out test data with Beam Search ($k=3$):

| Phase | Training Samples | Architecture (`D_MODEL` / `N_HEAD` / Layers) | Parameters | CER (Char Error Rate) | WER (Word Error Rate) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1: Sanity Check** | 1,000 | `128` / `8` / `3+3` (`dropout=0.0`) | ~1.1M | 0.0400 (4.00%) | — | Passed (Overfit Gate $< 5\%$) |
| **Phase 2: Prototype** | 100,000 | `256` / `8` / `3+3` (`best_model.pth`) | 4,071,576 | 0.2126 (21.26%) | 0.3980 (39.80%) | Passed (Gate $< 30\%$) |
| **Phase 3: Production** | **~2.27M (Cleaned)** | **`512` / `16` / `3+3` (`best_model_phase3.pth`)** | **12,858,520** | **0.0065 (0.65%)** | **0.0218 (2.18%)** | **Production Ready (99.35% Char Acc)** |

### 3.3 Beam Search Decoding Trade-off Analysis

Empirical evaluation across 500 fixed-seed test sentences (`random_state = 42`, Phase 2 baseline) comparing beam widths:

| Beam Width ($k$) | CER | WER | Avg. Latency / Sentence | Conclusion |
| :---: | :---: | :---: | :---: | :--- |
| **$k = 3$** | **0.1889** | **0.3035** | **~1.10 s** | **Optimal latency/accuracy sweet spot** |
| $k = 5$ | 0.1882 | 0.3030 | ~1.97 s | Negligible gain (-0.07% CER) for 1.8x latency |
| $k = 10$ | 0.1877 | 0.3029 | ~3.96 s | Diminishing return (-0.12% CER) for 3.6x latency |

---

## 4. Interactive Desktop Applications (GUI)

Both modules include standalone Tkinter desktop studios that dynamically register `misc/font/KhoaDau-Regular_v3.otf` via the Windows GDI API (`AddFontResourceExW`) without requiring manual system font installation:

1. **Module 1 Studio (`quoc_ngu_to_khoa_dau/inference.py` — *Khoa Đẩu Transliteration Studio v1.0*)**:
   - Real-time Quốc Ngữ $\rightarrow$ Khoa Đẩu rendering across `Rule-based`, `Dictionary`, `Shallow_Big`, `Big`, `Shallow_Small`, and `Small`.
   - Automatic dictionary filtering for foreign words and red glyph highlighting whenever a selected AI model diverges from the rule-based phonological ground truth.
   - Random real-world sentence sampler from `source.txt`.

2. **Module 2 Studio (`khoa_dau_to_quoc_ngu/inference.py` — *Khoa Đẩu to Quốc Ngữ AI Transliteration v2.0*)**:
   - **Round-Trip Double Translation**: Accepts Quốc Ngữ text, generates intermediate Khoa Đẩu script via `encode_safe`, and reconstructs Quốc Ngữ via the Seq2Seq Transformer (`beam_width = 3`) upon clicking **`⚡ Dịch`**.
   - **Architecture Auto-Detection & Hot-Swapping**: Inspects checkpoint metadata or `state_dict` tensor shapes on the fly, allowing instant dropdown switching between `best_model_phase3.pth` (`D_MODEL = 512`) and `best_model.pth` (`D_MODEL = 256`).
   - **Word-Level `SequenceMatcher` Diffing**: Aligns original and reconstructed sentences to underline substituted/inserted errors in **red** and inject explicit **`[ missing_word ]`** markers in **blue** without cascading alignment drift.

---

## 5. Repository Structure

```text
Applying NLP Techniques to Khoa Dau Transliteration/
├── requirements.txt                         # Verified Python dependencies
├── README.md                                # Project documentation
├── quoc_ngu_to_khoa_dau/                    # Module 1: Quoc Ngu -> Khoa Dau
│   ├── inference.py                         # Tkinter GUI Studio v1.0
│   ├── train.py                             # Unified CLI trainer for 1D CNN presets
│   ├── evaluate.py                          # Throughput & accuracy benchmark suite
│   ├── evaluate_mismatches.py               # Exhaustive 17,974-syllable hex diff checker
│   ├── data/                                # Normalized syllable lexicon, encoded CSV & benchmark text
│   ├── src/                                 # Rule-based engine, KhoaDauCNN models, data & parallel utils
│   └── weights/                             # Pretrained CNN weights (8 KB - 17 KB) & vocab.pth
├── khoa_dau_to_quoc_ngu/                    # Module 2: Khoa Dau -> Quoc Ngu
│   ├── inference.py                         # Tkinter GUI Studio v2.0 (Double Translation & Alignment)
│   ├── train.py                             # Seq2Seq Transformer trainer (AMP, Dynamic Padding, Resume)
│   ├── evaluate.py                          # Vectorized Beam Search evaluator (CER / WER)
│   ├── download_and_extract_hf.py           # Hugging Face corpus downloader -> Parquet shards
│   ├── data_maker.py                        # Modern-tone normalizer & synthetic pair generator
│   ├── prepare_final_dataset.py             # Deduplication & 90/5/5 train/val/test splitter
│   ├── Cloud_Notebook.ipynb                 # Google Colab A100 training notebook
│   ├── src/                                 # Seq2SeqTransformer, CharTokenizer, Dynamic DataLoader
│   └── weights/                             # Phase 2 & Phase 3 checkpoints + vocab.json
└── misc/                                    # Font & glyph visual assets
    ├── font/KhoaDau-Regular_v3.otf          # Custom PUA OpenType font (U+E000 - U+E026)
    ├── script_generator.py                  # Pillow PNG glyph exporter
    └── khoa_dau_script/                     # 36 rendered Khoa Dau character PNGs
```

---

## 6. Installation & Usage (Windows Command Prompt)

All commands below are formatted for **Windows 11 Command Prompt (`cmd.exe`)** from the project root directory.

### 6.1 Environment Setup

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
set PYTHONUTF8=1
```

*(Note: By default on Windows PyPI, `pip install -r requirements.txt` installs the stable CPU build of PyTorch. For cloud/CUDA GPU training, install the matching CUDA wheel from `https://download.pytorch.org/whl/cu126` prior to installing `requirements.txt`.)*

### 6.2 Module 1: Quốc Ngữ $\rightarrow$ Khoa Đẩu

Launch the interactive desktop studio:
```cmd
cd quoc_ngu_to_khoa_dau
..\.venv\Scripts\python.exe inference.py
```

Run throughput & accuracy benchmarks across Rule-based, Dictionary, and 1D CNN models:
```cmd
cd quoc_ngu_to_khoa_dau
set PYTHONUTF8=1
..\.venv\Scripts\python.exe evaluate.py
..\.venv\Scripts\python.exe evaluate_mismatches.py
```

Train a 1D CNN variant (`shallow_big`, `big`, `shallow_small`, or `small`):
```cmd
cd quoc_ngu_to_khoa_dau
set PYTHONUTF8=1
..\.venv\Scripts\python.exe train.py --model shallow_big --epochs 500 --batch_size 512 --lr 0.001
```

### 6.3 Module 2: Khoa Đẩu $\rightarrow$ Quốc Ngữ

Launch the v2.0 bidirectional evaluation studio (supports hot-swapping `best_model_phase3.pth` and `best_model.pth`):
```cmd
cd khoa_dau_to_quoc_ngu
..\.venv\Scripts\python.exe inference.py
```

Evaluate CER and WER on `data\test.csv` with Beam Search (`beam_width = 3`):
```cmd
cd khoa_dau_to_quoc_ngu
set PYTHONUTF8=1
..\.venv\Scripts\python.exe evaluate.py 3 best_model_phase3.pth
```

Build the corpus from scratch (optional — requires ~25 GB disk space) and train the Seq2Seq Transformer:
```cmd
set PYTHONUTF8=1
.venv\Scripts\python.exe khoa_dau_to_quoc_ngu\download_and_extract_hf.py
.venv\Scripts\python.exe khoa_dau_to_quoc_ngu\data_maker.py
.venv\Scripts\python.exe khoa_dau_to_quoc_ngu\prepare_final_dataset.py

cd khoa_dau_to_quoc_ngu
..\.venv\Scripts\python.exe train.py --limit 100000
```
