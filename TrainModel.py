import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from SequenceData import MIDITokenDataset  # Custom Dataset class

# --- Hyperparameters ---
SEQ_LEN = 100        
BATCH_SIZE = 64       
EPOCHS = 10           
EMBED_SIZE = 128      
HIDDEN_SIZE = 256     
NUM_LAYERS = 2        
LR = 0.001           

# --- Load token vocabulary ---
with open("vocab.json", "r") as f:
    token_to_idx = json.load(f)

vocab_size = len(token_to_idx)  # total number of unique tokens

# --- Load dataset and wrap it in a DataLoader ---
dataset = MIDITokenDataset("tokens.jsonl", token_to_idx, seq_len=SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Define the LSTM-based language model ---
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # turn token indices into vectors
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)  # predict logits over all tokens

    def forward(self, x):
        x = self.embed(x)          # shape: (batch, seq_len) → (batch, seq_len, embed_size)
        out, _ = self.lstm(x)      # shape: (batch, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])  # use output at last timestep → (batch, vocab_size)
        return out

# --- Set device to GPU if available ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialize and move model to device ---
model = LSTMModel(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)

# --- Define loss and optimizer ---
criterion = nn.CrossEntropyLoss()                  # loss between predicted logits and actual token index
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Training loop ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x_batch, y_batch in dataloader:
        # Move data to GPU if available
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(x_batch)       # shape: (batch_size, vocab_size)
        loss = criterion(output, y_batch)

        # Backward pass and parameter update
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# --- Save the trained model ---
torch.save(model.state_dict(), "lstm_model.pth")
print("✅ Model saved to lstm_model.pth")
