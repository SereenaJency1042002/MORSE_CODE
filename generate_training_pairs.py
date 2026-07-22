import csv
import os
import random

random.seed(42)

LABEL_FILES = [
    "audio_files/training data set/labels.csv",
    "augmented_data/train/labels.csv",
    "augmented_data/val/labels.csv",
]

OUTPUT_FILE = "morse_training_pairs.csv"

PROSIGNS = ['CQ', 'DE', 'AR', 'SK', 'BK', 'KN', 'AS', 'TU', 'BT']

def corrupt_question_marks(text: str, rate: float = 0.2) -> str:
    tokens = text.split()
    result = []
    for token in tokens:
        if random.random() < rate and token not in PROSIGNS:
            if random.random() < 0.5:
                result.append('?')
            else:
                chars = list(token)
                n = max(1, int(len(chars) * 0.4))
                for _ in range(n):
                    idx = random.randint(0, len(chars)-1)
                    chars[idx] = '?'
                result.append(''.join(chars))
        else:
            result.append(token)
    return ' '.join(result)

def corrupt_timing_errors(text: str) -> str:
    replacements = {
        'TU':  'TA',
        '5NN': '5N',
        'CQ':  random.choice(['CQ', 'CQCQ']),
        'DE':  random.choice(['DE', 'DEDE']),
        'SK':  random.choice(['SK', 'SKSK']),
    }
    tokens = text.split()
    result = []
    for token in tokens:
        if token in replacements and random.random() < 0.6:
            result.append(replacements[token])
        else:
            result.append(token)
    return ' '.join(result)

def corrupt_duplicates(text: str) -> str:
    tokens = text.split()
    result = []
    for token in tokens:
        result.append(token)
        if token in PROSIGNS and random.random() < 0.3:
            extra = random.randint(1, 3)
            result.extend([token] * extra)
    return ' '.join(result)

def corrupt_heavy(text: str, rate: float = 0.5) -> str:
    tokens = text.split()
    result = []
    for token in tokens:
        if random.random() < rate:
            result.append('?')
        else:
            result.append(token)
    return ' '.join(result)

def corrupt_combined(text: str) -> str:
    text = corrupt_timing_errors(text)
    text = corrupt_question_marks(text, rate=0.15)
    text = corrupt_duplicates(text)
    return text

CORRUPTION_FUNCTIONS = [
    corrupt_question_marks,
    corrupt_timing_errors,
    corrupt_duplicates,
    corrupt_combined,
    corrupt_heavy,
]

def load_all_labels() -> list:
    texts = set()
    for path in LABEL_FILES:
        if not os.path.exists(path):
            print(f"  [SKIP] Not found: {path}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get('expected_text', '').strip().upper()
                if text and len(text) > 3:
                    texts.add(text)
        print(f"  Loaded: {path}")
    return list(texts)

def generate():
    print("Generating Morse training pairs...")
    print("-" * 50)
    correct_texts = load_all_labels()
    print(f"Unique correct texts: {len(correct_texts)}")
    pairs = []
    for correct in correct_texts:
        pairs.append({
            "corrupt_text": correct,
            "correct_text": correct,
            "corruption":   "none"
        })
        for func in CORRUPTION_FUNCTIONS:
            corrupted = func(correct)
            if corrupted != correct:
                pairs.append({
                    "corrupt_text": corrupted,
                    "correct_text": correct,
                    "corruption":   func.__name__
                })
    random.shuffle(pairs)
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=["corrupt_text", "correct_text", "corruption"]
        )
        writer.writeheader()
        writer.writerows(pairs)
    print(f"\nTotal pairs generated: {len(pairs)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nSample pairs:")
    for pair in pairs[:3]:
        print(f"  IN:   {pair['corrupt_text'][:60]}")
        print(f"  OUT:  {pair['correct_text'][:60]}")
        print(f"  TYPE: {pair['corruption']}")
        print()

if __name__ == "__main__":
    generate()
