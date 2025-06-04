# Import necessary PyTorch modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import time
from SequenceData import MIDITokenDataset  # Import custom dataset class

# --- Hyperparameters ---
SEQ_LEN = 512        # Length of input token sequences
BATCH_SIZE = 32      # Number of samples per training batch
EPOCHS = 30          # Number of full passes through the training data
EMBED_SIZE = 128     # Embedding dimension for token indices
HIDDEN_SIZE = 256    # Hidden state size of LSTM
NUM_LAYERS = 2       # Number of stacked LSTM layers
LR = 0.001           # Learning rate for the optimizer

# --- Load token vocabulary from file ---
with open("vocab.json", "r") as f:
    token_to_idx = json.load(f)

vocab_size = len(token_to_idx)  # Total number of unique tokens in vocabulary

# --- Initialize dataset and data loader ---
dataset = MIDITokenDataset("tokens.jsonl", token_to_idx, seq_len=SEQ_LEN)
print(f"Dataset size: {len(dataset)} samples")  # Print how many samples are in the dataset

# DataLoader will load batches of token sequences during training
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,       # Shuffle dataset each epoch
    pin_memory=False,   # Not using pinned memory (safer for Windows)
    num_workers=0       # Single-threaded loading (safer for Windows)
)

# --- Define the LSTM model architecture ---
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # Converts token indices to vectors
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # LSTM layers
        self.fc = nn.Linear(hidden_size, vocab_size)  # Final layer projects to vocabulary size

    def forward(self, x):
        x = self.embed(x)            # (batch, seq_len) -> (batch, seq_len, embed_size)
        out, _ = self.lstm(x)        # Output from LSTM: (batch, seq_len, hidden_size)
        out = self.fc(out[:, -1, :]) # Use only the final time step output -> (batch, vocab_size)
        return out

# --- Setup training device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")  # Confirm GPU identity

# Instantiate the model and move it to the correct device
model = LSTMModel(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Standard loss for classification
optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # Adam optimizer

# --- Main training loop ---
for epoch in range(EPOCHS):
    model.train()  # Set model to training mode
    total_loss = 0  # Accumulate loss for averaging
    epoch_start = time.time()  # Start timer for benchmarking

    # Loop over all batches
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        # Move input and target to GPU if available
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()     # Clear gradients
        output = model(x_batch)   # Forward pass through the model
        loss = criterion(output, y_batch)  # Compute loss
        loss.backward()           # Backpropagate gradients
        optimizer.step()          # Update model parameters

        total_loss += loss.item()  # Accumulate loss for epoch average

    # Compute average loss and epoch duration
    avg_loss = total_loss / len(dataloader)
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f} sec")

# --- Save trained model weights to file ---
torch.save(model.to("cpu").state_dict(), "lstm_model.pth")  # Move model to CPU before saving
print("Model saved to lstm_model.pth")
