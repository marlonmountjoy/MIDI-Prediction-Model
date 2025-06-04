import torch
import torch.nn as nn
import json
from SequenceData import MIDITokenDataset

# --- Generation Settings ---
SEED_TOKENS = [
    "note_on_60", "note_on_64", "note_on_67",  # C major chord: C-E-G
    "time_shift_240",
    "note_off_60", "note_off_64", "note_off_67"
]
GENERATE_LENGTH = 10000 
MODEL_PATH = "lstm_model.pth"
OUTPUT_FILE = "generatedTokens.json"

# --- Load vocab ---
with open("vocab.json", "r") as f:
    token_to_idx = json.load(f) # Token map
idx_to_token = {v: k for k, v in token_to_idx.items()} # Reverse map
vocab_size = len(token_to_idx)

# --- LSTM model definition ---
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size) # Token index to vector
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True) # LSTM layers
        self.fc = nn.Linear(hidden_size, vocab_size) # Output layer

    def forward(self, x, hidden=None):
        x = self.embed(x) # Convert input indicies
        out, hidden = self.lstm(x, hidden) # Pass through LSTM layers
        out = self.fc(out[:, -1, :]) # Get output
        return out, hidden

# --- Load trained model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(vocab_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device)) # Load trained weights
model.eval().to(device)

# --- Convert seed to tensor ---
input_seq = [token_to_idx[t] for t in SEED_TOKENS if t in token_to_idx]
input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

# --- Generate new tokens ---
generated = SEED_TOKENS.copy()
hidden = None

# Turn off gradient tracking for generation
with torch.no_grad():
    for _ in range(GENERATE_LENGTH):
        output, hidden = model(input_tensor, hidden)  # Get model prediction and updated hidden state
        prob = torch.softmax(output[0], dim=0)  # Convert logits to probabilities
        idx = torch.multinomial(prob, num_samples=1).item()  # Sample from distribution

        token = idx_to_token[idx]  # Convert index back to token
        generated.append(token)  # Add token to the generated sequence

        # Prepare next input using the latest generated token
        input_tensor = torch.tensor([[idx]], dtype=torch.long).to(device)

# --- Print and save results ---
print("Generated sequence preview:")
print(generated[:50])  # preview only

with open(OUTPUT_FILE, "w") as f:
    json.dump(generated, f)

print(f"Generated sequence saved to {OUTPUT_FILE}")