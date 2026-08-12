# 🦅 EagleSFT — Bilingual (EN-RU) Dataset Pipeline & LLM Fine-Tuning

A 536K-example bilingual (Russian-English) supervised fine-tuning dataset, built end-to-end
from raw question collection through translation, quality filtering, and categorization —
plus a LoRA fine-tuning pipeline that trains Qwen2.5 on it for category prediction and
text generation.



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

**annotations_creators:** machine-generated
**language:** ru, en
**multilinguality:** bilingual
**pretty_name:** EagleSFT
**size_categories:** 100K<n<1M
**source_datasets:** original
**task_categories:** text-generation, text-classification
**task_ids:** language-modeling
**tags:** synthetic

**configs:**

| config_name | split | path |
|---|---|---|
| train | en | en_train.jsonl.zst |
| train | ru | ru_train.jsonl.zst |

### Dataset Summary

This dataset contains 536,231 pairs of human questions and machine-generated responses
intended for supervised fine-tuning (SFT) of large language models. The dataset includes
both Russian and English content, with linked IDs allowing for cross-lingual analysis. It
was created by processing an initial collection of 739,732 human questions posed to LLMs,
predominantly in Russian (about 99%) with a small portion in English (about 1%).

The main topic of questions is education (various tasks and questions on school and
university programs), but the dataset also includes everyday, non-serious, and humorous
questions, reflecting the natural, non-synthetic origin of the data.

### Languages

The dataset is bilingual:
- Russian (ru)
- English (en)

### Dataset Structure

This dataset includes the following fields:
- `id`: Unique identifier linking corresponding entries in Russian and English (string)
- `category`: Machine-generated category label for the question (string)
- `messages`: Array containing conversation elements:
  - `role`: Either "user" for questions or "assistant" for responses (string)
  - `content`: The text content of the question or response (string)

### Data Splits

The dataset has two splits:
- Russian: 536,231 question-response pairs
- English: 536,231 question-response pairs

These splits contain corresponding content with matching IDs, though not all content is
strictly translated — some data was originally in English or Russian.

### Data Creation Process

1. **Collection**: Initial collection of 739,732 human questions, primarily in Russian
   (99%) with some English (1%)
2. **Language sorting**: Questions were separated by language (Russian/English) and
   deduplicated
3. **ID assignment**: Each unique question was assigned a UUID
4. **Translation**: Using Mistral-Small-3.1-24B-Instruct-2503:
   - Russian questions were translated to English
   - English questions were translated to Russian
5. **Quality filtering**:
   - Translations were checked for Cyrillic characters
   - Failed translations were reprocessed with adjusted prompts/temperature
   - Further deduplication was performed
6. **Response generation**: Questions were processed through
   [Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)
   to generate model responses
7. **Final filtering**: Additional quality checks to remove duplicates and model
   repetitions
8. **Categorization**: Question categories were generated using
   [Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)
9. **Re-indexing**: Remaining 536,231 pairs were re-indexed to avoid gaps in data
   numbering

### License (dataset)

This dataset is dedicated to the public domain under the Creative Commons Zero (CC0)
license. This means you can:
- Use it for any purpose, including commercial projects.
- Modify it however you like.
- Distribute it without asking permission.

No attribution is required, but it's always appreciated!

CC0 license: https://creativecommons.org/publicdomain/zero/1.0/deed.en

To learn more about CC0, visit the Creative Commons website:
https://creativecommons.org/publicdomain/zero/1.0/

### Dataset Curators

Nawfal Abdul Malick, and team (USI NLP course group project).

</details>

## License

- **Code**: MIT
- **Dataset**: CC0-1.0 — public domain, no attribution required
