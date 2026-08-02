# 🦅 EagleSFT — Bilingual (EN-RU) Dataset Pipeline & LLM Fine-Tuning

A 536K-example bilingual (Russian-English) supervised fine-tuning dataset, built end-to-end
from raw question collection through translation, quality filtering, and categorization —
plus a LoRA fine-tuning pipeline that trains Qwen2.5 on it for category prediction and
text generation.

Originally a group project for the NLP course at USI Lugano.

## What's in this repo

| File | What it does |
|---|---|
| `prepare_dataset.py` | Data pipeline: language detection, deduplication, translation (via Mistral-Small-3.1-24B), quality filtering, and category labeling |
| `fine_tune.py` | LoRA fine-tuning of Qwen2.5 on EagleSFT for category prediction + text generation |
| `inference.ipynb` | Runs inference with the fine-tuned adapter |
| `Project.ipynb` | End-to-end notebook walkthrough (dataset prep → training → evaluation) |
| `qwen2.5_lora/final/` | Trained LoRA adapter weights |

## Pipeline overview

1. **Collection**: 739,732 raw questions (99% Russian, 1% English)
2. **Language sorting & deduplication**
3. **Translation**: cross-lingual pairs generated via Mistral-Small-3.1-24B-Instruct, with
   Cyrillic-character validation and reprocessing of failed translations
4. **Response generation**: model responses generated via Mistral-Small-3.1-24B-Instruct
5. **Categorization**: category labels generated via the same model
6. **Final filtering & re-indexing**: 536,231 final pairs, deduplicated and repetition-checked
7. **Fine-tuning**: Qwen2.5 trained via LoRA on the resulting dataset for category
   prediction and text generation

## Running it

```bash
pip install -r requirements.txt
python prepare_dataset.py        # rebuilds the dataset from raw sources (optional — dataset is included)
python fine_tune.py              # LoRA fine-tune Qwen2.5 on EagleSFT
```

Then open `inference.ipynb` to run the fine-tuned model on new inputs.

## Results

<!-- Fill in with your actual numbers — category classification accuracy/F1,
generation quality metrics, or a few example before/after outputs. Even 3-4
sample generations with the base vs. fine-tuned model side by side would be
convincing here. -->

## Dataset card: EagleSFT

<details>
<summary>Full dataset card (click to expand)</summary>

[... keep your existing dataset card content here — annotations_creators, language,
size_categories, data fields, splits, license, etc. exactly as it is now ...]

</details>

---
annotations_creators:
- machine-generated
language:
- ru
- en
multilinguality:
- bilingual
pretty_name: EagleSFT
size_categories:
- 100K<n<1M
source_datasets:
- original
task_categories:
- text-generation
- text-classification
task_ids:
- language-modeling
tags:
- synthetic
configs:
- config_name: train
  data_files:
  - split: en
    path: "en_train.jsonl.zst"
  - split: ru
    path: "ru_train.jsonl.zst"
license: cc0-1.0
---

## License

- **Code**: MIT
- **Dataset**: CC0-1.0 — public domain, no attribution required