import json
import torch
import numpy as np
import os
from TrainModel import LSTMModel
from SequenceData import Vocabulary

# Load the trained model
model = LSTMModel(input_size=512, hidden_size=256, output_size=512)
model.load_state_dict(torch.load("trained_model.pt", map_location=torch.device("cpu")))
model.eval()

# Load the vocabulary
vocab = Vocabulary.load("vocab.json")
idx_to_token = vocab.idx_to_token
token_to_idx = vocab.token_to_idx

# Generation settings
start_sequence = ["note_on_60", "time_shift_120", "note_off_60"]
generate_length = 1000

# Convert start sequence to indices
input_seq = [token_to_idx[token] for token in start_sequence]
input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)  # shape: (1, seq_len)

# Generate token sequence
generated = input_seq.copy()
with torch.no_grad():
    hidden = None
    for _ in range(generate_length):
        output, hidden = model(input_tensor[:, -1].unsqueeze(1), hidden)
        probs = torch.softmax(output[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)
        input_tensor = torch.tensor([generated[-len(start_sequence):]], dtype=torch.long)

# Convert back to tokens
tokens = [idx_to_token[idx] for idx in generated]

# Save to ./output/generatedTokens.json
output_path = os.path.join("output", "generatedTokens.json")
os.makedirs("output", exist_ok=True)
with open(output_path, "w") as f:
    json.dump(tokens, f, indent=2)

print(f"Tokens generated and saved to {output_path}")
