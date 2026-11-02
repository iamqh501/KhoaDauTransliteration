# Applying NLP Techniques to Khoa Dau Transliteration

A bidirectional transliteration system between Vietnamese **Quốc Ngữ** (Latin script) and **Khoa Đẩu** script, rendered via a 36-glyph Unicode Private Use Area font (`U+E000`–`U+E026`).

Transliteration between the two scripts is asymmetric:
- **Quốc Ngữ $\rightarrow$ Khoa Đẩu** is deterministic: tone marks are dropped, final consonants take terminal forms, specific vowels shift before the onset, and homophonous graphemes (such as `c/k/q`, `d/gi/r`, `ch/tr`, `ng/ngh`) merge into shared glyphs.
- **Khoa Đẩu $\rightarrow$ Quốc Ngữ** is one-to-many: reconstructing tone marks and selecting the right Latin spelling requires sentence-level context.

## Methodology

### 1. Quốc Ngữ to Khoa Đẩu (`quoc_ngu_to_khoa_dau`)
- **Dictionary Filtering**: Input tokens are checked against a lexicon of 17,974 normalized Vietnamese syllables. Non-Vietnamese words, numbers, and symbols bypass conversion and remain unchanged.
- **Transliteration Engines**: Implemented and compared across three methods:
  1. Phonological rule-based conversion.
  2. Precomputed syllable dictionary lookup on CPU and GPU tensors.
  3. Character-level 1D Convolutional Neural Networks (`KhoaDauCNN`, fixed 7-character window, trained without dropout).

### 2. Khoa Đẩu to Quốc Ngữ (`khoa_dau_to_quoc_ngu`)
- **Dataset**: Built from `BlossomsAI/vietnamese-corpus` (~4M raw sentences). After stripping non-Latin scripts, normalizing to modern Vietnamese tone placement (`oà, oè, uý`), padding punctuation, and removing duplicates, the final corpus contains ~2.27M unique sentence pairs split 90% train / 5% validation / 5% test.
- **Model & Decoding**: An Encoder-Decoder **Seq2Seq Transformer** operating on a 152-character vocabulary (covering Khoa Đẩu glyphs, Vietnamese Latin letters, digits, and punctuation so foreign tokens can be copied character-by-character). Training uses in-memory pre-tokenization, per-batch dynamic padding, and FP16 mixed precision. Inference uses batched Beam Search ($k = 3$).

## Experimental Results

### Quốc Ngữ $\rightarrow$ Khoa Đẩu
Accuracy is measured across all 17,974 Vietnamese syllables; latency is measured on a 204,758-word benchmark text (`quoc_ngu_to_khoa_dau/data/source.txt`).

| Method | Configuration | Syllable Accuracy | Latency (ms) | Parameters | Weight Size |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Dictionary (GPU) | Tensor indexing | 100.00% | ~399 | — | 315.26 KB |
| Dictionary (CPU) | Single-process hash map | 100.00% | ~413 | — | 315.26 KB |
| `KhoaDauCNN` (`shallow_big`) | 2 layers, kernel 5, dims 8/16 | 100.00% | ~885 | 3,326 | 16.70 KB |
| `KhoaDauCNN` (`big`) | 3 layers, kernel 3, dims 8/16 | 99.93% | ~1,253 | 3,342 | 17.23 KB |
| Rule-Based (`encode_safe`) | Single-process | 100.00% | ~1,295 | — | 7.11 KB |
| `KhoaDauCNN` (`shallow_small`) | 2 layers, kernel 5, dims 4/8 | 75.28% | — | 1,202 | 8.37 KB |
| `KhoaDauCNN` (`small`) | 3 layers, kernel 3, dims 4/8 | 66.73% | — | 1,210 | 8.81 KB |

*(On Windows, multi-process CPU execution via `ProcessPoolExecutor` measured ~6,600 ms on the same text due to process spawning and serialization overhead.)*

### Khoa Đẩu $\rightarrow$ Quốc Ngữ
Evaluated on the held-out test set using Beam Search ($k = 3$). Increasing beam width from $k = 3$ to $k = 10$ on a 500-sentence sample reduced CER by 0.0012 (0.1889 $\rightarrow$ 0.1877) while increasing decoding time from ~1.10 s to ~3.96 s per sentence.

| Model Checkpoint | Training Pairs | `d_model` / Heads / Layers | Parameters | Weight Size | CER | WER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `best_model.pth` (Phase 2) | 100,000 | 256 / 8 / 3+3 | 4,071,576 | 20.45 MB | 0.2126 | 0.3980 |
| `best_model_phase3.pth` (Phase 3) | ~2.04M | 512 / 16 / 3+3 | 12,858,520 | 58.85 MB | 0.0065 | 0.0218 |

## Setup & Usage (Windows `cmd.exe`)

### 1. Install Dependencies
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set PYTHONUTF8=1
```

### 2. Desktop Applications (Tkinter GUI)
Both apps load `misc\font\KhoaDau-Regular_v3.otf` automatically at runtime.

- **Quốc Ngữ $\rightarrow$ Khoa Đẩu GUI** (switches between rule-based, dictionary, and CNN models; highlights mismatches against rule-based output):
  ```cmd
  cd quoc_ngu_to_khoa_dau
  ..\.venv\Scripts\python.exe inference.py
  ```

- **Khoa Đẩu $\rightarrow$ Quốc Ngữ GUI** (converts Quốc Ngữ input to Khoa Đẩu, runs Seq2Seq reconstruction, supports checkpoint switching, and marks word differences/omissions):
  ```cmd
  cd khoa_dau_to_quoc_ngu
  ..\.venv\Scripts\python.exe inference.py
  ```

### 3. Evaluation & Training CLI
```cmd
REM Module 1 benchmarks and training
cd quoc_ngu_to_khoa_dau
..\.venv\Scripts\python.exe evaluate.py
..\.venv\Scripts\python.exe evaluate_mismatches.py
..\.venv\Scripts\python.exe train.py --model shallow_big

REM Module 2 evaluation and training
cd ..\khoa_dau_to_quoc_ngu
..\.venv\Scripts\python.exe evaluate.py 3 best_model_phase3.pth
..\.venv\Scripts\python.exe train.py --limit 100000
```
