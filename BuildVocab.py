import json
from collections import Counter
from pathlib import Path

# Load token file from current working directory
token_file = Path("tokens.jsonl")
token_sequences = []

# Load each token sequence from file
with token_file.open() as f:
    for line in f:
        tokens = json.loads(line)
        token_sequences.append(tokens)

# Build vocabulary
all_tokens = [token for seq in token_sequences for token in seq]
token_freq = Counter(all_tokens)
vocab = sorted(token_freq.keys())

# Create mappings
token_to_idx = {token: idx for idx, token in enumerate(vocab)}
idx_to_token = {idx: token for token, idx in token_to_idx.items()}

# Save vocab for reuse (optional)
with open("vocab.json", "w") as f:
    json.dump(token_to_idx, f)

# Show basic info
print(f"\u2705✅ Vocabulary size: {len(vocab)}")
print(f"\U0001F9E0 Sample tokens: {list(token_to_idx.items())[:10]}")
