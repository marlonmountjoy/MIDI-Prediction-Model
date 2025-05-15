import torch
import torch.nn as nn
import json
from SequenceData import MIDITokenDataset

# --- Generation settings ---
SEED_TOKENS = ["note_on_60", "time_shift_120", "note_off_60"]
GENERATE_LENGTH = 100  # How many new tokens to generate
MODEL_PATH = "lstm_model.pth"

# --- Load vocab ---
with open("vocab.json", "r") as f:
    token_to_idx = json.load(f)
idx_to_token = {v: k for k, v in token_to_idx.items()}
vocab_size = len(token_to_idx)

# --- LSTM model class ---
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # only return final timestep
        return out, hidden

# --- Load model and weights ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(vocab_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

# --- Convert seed to tensor ---
input_seq = [token_to_idx[t] for t in SEED_TOKENS if t in token_to_idx]
input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

# --- Generate new tokens ---
generated = SEED_TOKENS.copy()
hidden = None

for _ in range(GENERATE_LENGTH):
    output, hidden = model(input_tensor, hidden)
    prob = torch.softmax(output[0], dim=0)
    idx = torch.multinomial(prob, num_samples=1).item()

    token = idx_to_token[idx]
    generated.append(token)

    # Update input tensor with most recent token
    input_tensor = torch.tensor([[idx]], dtype=torch.long).to(device)

# --- Print generated tokens ---
print("🪄 Generated sequence:")
print(generated)

