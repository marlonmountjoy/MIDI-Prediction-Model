import sys
import os
import torch
from SequenceData import MIDITokenDataset
from torch.utils.data import DataLoader
from TrainModel import LSTMModel
import json
import torch.nn as nn

# Parse dataset directory from command-line
if len(sys.argv) < 2:
    print("Usage: python train_model.py <dataset_path>")
    sys.exit(1)

dataset_path = sys.argv[1]

# Load vocabulary
with open("vocab.json", "r") as f:
    vocab = json.load(f)

vocab_size = len(vocab)

# Load dataset
dataset = MIDITokenDataset("tokens.jsonl", vocab)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop
EPOCHS = 5
for epoch in range(EPOCHS):
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "lstm_model.pth")
print("💾 Model saved as lstm_model.pth")