import json
import random
import zstandard as zstd
from collections import Counter, defaultdict
from tqdm import tqdm
import io

# =========================
# CONFIG
# =========================
EN_FILE = "./EagleSFT/en_train.jsonl.zst"
RU_FILE = "./EagleSFT/ru_train.jsonl.zst"

TOP_N_CATEGORIES = 10
SAMPLES_PER_CATEGORY = 5000
TRANSLATION_RATIO = 0.25  # 25% of samples get translation variants
OUTPUT_FILE = "finetune_lora_multilingual_150k.jsonl"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# =========================
# HELPERS
# =========================
def stream_jsonl_zst(path):
    """Stream decompress and parse JSONL from zstandard file"""
    with open(path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                yield json.loads(line)

# =========================
# STEP 1: FULL SCAN — CATEGORIES
# =========================
print("🔍 Full scan: counting categories (EN + RU)...")

category_counter = Counter()

for file in [EN_FILE, RU_FILE]:
    for sample in tqdm(stream_jsonl_zst(file), desc=f"Scanning {file}"):
        category_counter[sample["category"]] += 1

top_categories = [c for c, _ in category_counter.most_common(TOP_N_CATEGORIES)]

print("\n✅ Top categories (exact):")
for i, c in enumerate(top_categories, 1):
    print(f"{i:2d}. {c:25s} {category_counter[c]:,}")

# =========================
# STEP 2: LOAD DATA WITH ID MAPPING
# =========================
print("\n📦 Loading data with ID mapping for parallel translations...")

# Store samples by ID for matching
en_samples = {}  # id -> sample
ru_samples = {}  # id -> sample

# Category buckets for sampling
en_buckets = defaultdict(list)  # (category) -> [ids]
ru_buckets = defaultdict(list)  # (category) -> [ids]

print("Loading English samples...")
for sample in tqdm(stream_jsonl_zst(EN_FILE), desc="Loading EN"):
    cat = sample["category"]
    sid = sample["id"]
    
    if cat in top_categories:
        en_samples[sid] = sample
        if len(en_buckets[cat]) < SAMPLES_PER_CATEGORY:
            en_buckets[cat].append(sid)

print("Loading Russian samples...")
for sample in tqdm(stream_jsonl_zst(RU_FILE), desc="Loading RU"):
    cat = sample["category"]
    sid = sample["id"]
    
    if cat in top_categories:
        ru_samples[sid] = sample
        if len(ru_buckets[cat]) < SAMPLES_PER_CATEGORY:
            ru_buckets[cat].append(sid)

# =========================
# STEP 3: FIND PARALLEL PAIRS
# =========================
print("\n🔗 Finding parallel translation pairs...")

parallel_pairs = defaultdict(list)  # category -> [(en_id, ru_id)]

for cat in top_categories:
    en_ids = set(en_buckets[cat])
    ru_ids = set(ru_buckets[cat])
    
    # Find matching IDs (parallel translations)
    common_ids = en_ids & ru_ids
    
    parallel_pairs[cat] = list(common_ids)
    print(f"{cat:25s} - Parallel pairs: {len(parallel_pairs[cat]):,}")

# =========================
# STEP 4: BUILD TRAINING DATA
# =========================
print("\n🛠 Building training samples...")

training_data = []
translation_samples = 0

for cat in top_categories:
    common_ids = parallel_pairs[cat]
    
    # Shuffle to randomize which ones get translation tasks
    random.shuffle(common_ids)
    
    for idx, sid in enumerate(common_ids):
        # Get parallel samples
        en_sample = en_samples[sid]
        ru_sample = ru_samples[sid]
        
        # Extract questions and answers
        en_q = en_sample["messages"][0]["content"]
        en_a = en_sample["messages"][1]["content"]
        ru_q = ru_sample["messages"][0]["content"]
        ru_a = ru_sample["messages"][1]["content"]
        
        # 1. Add English Q&A
        training_data.append({
            "id": f"{sid}_en",
            "category": cat,
            "language": "en",
            "task_type": "qa",
            "messages": [
                {"role": "user", "content": en_q},
                {"role": "assistant", "content": en_a}
            ]
        })
        
        # 2. Add Russian Q&A
        training_data.append({
            "id": f"{sid}_ru",
            "category": cat,
            "language": "ru",
            "task_type": "qa",
            "messages": [
                {"role": "user", "content": ru_q},
                {"role": "assistant", "content": ru_a}
            ]
        })
        
        # 3. Add translation tasks (25% of pairs)
        if idx < int(len(common_ids) * TRANSLATION_RATIO):
            # EN question -> RU question
            training_data.append({
                "id": f"{sid}_trans_en_ru_q",
                "category": cat,
                "language": "en->ru",
                "task_type": "translation",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Translate the following question to Russian:\n\n{en_q}"
                    },
                    {"role": "assistant", "content": ru_q}
                ]
            })
            
            # RU question -> EN question
            training_data.append({
                "id": f"{sid}_trans_ru_en_q",
                "category": cat,
                "language": "ru->en",
                "task_type": "translation",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Translate the following question to English:\n\n{ru_q}"
                    },
                    {"role": "assistant", "content": en_q}
                ]
            })
            
            # EN answer -> RU answer
            training_data.append({
                "id": f"{sid}_trans_en_ru_a",
                "category": cat,
                "language": "en->ru",
                "task_type": "translation",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Translate the following answer to Russian:\n\n{en_a}"
                    },
                    {"role": "assistant", "content": ru_a}
                ]
            })
            
            # RU answer -> EN answer
            training_data.append({
                "id": f"{sid}_trans_ru_en_a",
                "category": cat,
                "language": "ru->en",
                "task_type": "translation",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Translate the following answer to English:\n\n{ru_a}"
                    },
                    {"role": "assistant", "content": en_a}
                ]
            })
            
            translation_samples += 4

# Shuffle all training data
random.shuffle(training_data)

# =========================
# STEP 5: SAVE JSONL
# =========================
print(f"\n💾 Saving to {OUTPUT_FILE} ...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for s in training_data:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print("\n✅ DONE")
print(f"Total samples: {len(training_data):,}")
print(f"  - Q&A samples: {len(training_data) - translation_samples:,}")
print(f"  - Translation samples: {translation_samples:,}")
print(f"Categories: {TOP_N_CATEGORIES}")
print(f"Translation ratio: {int(TRANSLATION_RATIO * 100)}%")
print(f"Output file: {OUTPUT_FILE}")

# Print detailed statistics
print("\n📊 Task type distribution:")
task_dist = Counter([s["task_type"] for s in training_data])
for task, count in task_dist.items():
    print(f"  {task:15s} {count:,}")

print("\n📊 Category distribution:")
cat_dist = Counter([s["category"] for s in training_data])
for cat, count in cat_dist.most_common():
    print(f"  {cat:25s} {count:,}")

print("\n🌍 Language distribution:")
lang_dist = Counter([s["language"] for s in training_data])
for lang, count in sorted(lang_dist.items()):
    print(f"  {lang:10s} {count:,}")