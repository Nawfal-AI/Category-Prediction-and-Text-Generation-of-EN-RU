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

# Dataset Card for 🦅 EagleSFT

### Dataset Summary

This dataset contains 536,231 pairs of human questions and machine-generated responses intended for supervised fine-tuning (SFT) of large language models. The dataset includes both Russian and English content, with linked IDs allowing for cross-lingual analysis. It was created by processing an initial collection of 739,732 human questions posed to LLMs, predominantly in Russian (about 99%) with a small portion in English (about 1%).

The main topic of questions is education (various tasks and questions on school and university programs), but the dataset also includes everyday, non-serious, and humorous questions, reflecting the natural, non-synthetic origin of the data.

### Languages

The dataset is bilingual:
- Russian (ru)
- English (en)

## Dataset Structure

### Data Fields

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

These splits contain corresponding content with matching IDs, though not all content is strictly translated - some data was originally in English or Russian.

### Data Creation Process

1. Collection: Initial collection of 739,732 human questions, primarily in Russian (99%) with some English (1%)
2. Language sorting: Questions were separated by language (Russian/English) and deduplicated
3. ID assignment: Each unique question was assigned a UUID
4. Translation: Using [Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503):
   - Russian questions were translated to English
   - English questions were translated to Russian
5. Quality filtering:
   - Translations were checked for Cyrillic characters
   - Failed translations were reprocessed with adjusted prompts/temperature
   - Further deduplication was performed
6. Response generation: Questions were processed through [Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) to generate model responses
7. Final filtering: Additional quality checks to remove duplicates and model repetitions
8. Categorization: Question categories were generated using [Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)
9. Re-indexing: Remaining 536,231 pairs were re-indexed to avoid gaps in data numbering

### License

This dataset is dedicated to the public domain under the Creative Commons Zero (CC0) license. This means you can:

* Use it for any purpose, including commercial projects.
* Modify it however you like.
* Distribute it without asking permission.

No attribution is required, but it's always appreciated!

CC0 license: https://creativecommons.org/publicdomain/zero/1.0/deed.en

To learn more about CC0, visit the Creative Commons website: https://creativecommons.org/publicdomain/zero/1.0/

### Dataset Curators

- [nyuuzyou](https://ducks.party)
