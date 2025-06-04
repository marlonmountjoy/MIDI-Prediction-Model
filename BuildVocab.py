# Import required modules
import json
from collections import Counter
from pathlib import Path

# Define path to the token file (assumes it's in the current working directory)
token_file = Path("tokens.jsonl")

# Initialize a Counter to accumulate token frequencies efficiently
token_freq = Counter()

# Open the token file and update the frequency counter line-by-line
with token_file.open() as f:
    for line in f:
        tokens = json.loads(line)           # Parse each line as a list of tokens
        token_freq.update(tokens)           # Add token frequencies to the Counter

# Sort tokens alphabetically for consistent vocab indexing
vocab = sorted(token_freq.keys())

# Create a token-to-index mapping
token_to_idx = {token: idx for idx, token in enumerate(vocab)}

# Create an index-to-token reverse mapping (not saved, but useful for debugging)
idx_to_token = {idx: token for token, idx in token_to_idx.items()}

# Save the token-to-index dictionary to a JSON file
with open("vocab.json", "w") as f:
    json.dump(token_to_idx, f)

# Print summary information
print(f"\u2705 Vocabulary size: {len(vocab)}")  # Total number of unique tokens
print(f"\U0001F9E0 Sample tokens: {list(token_to_idx.items())[:10]}")  # Show first few mappings
