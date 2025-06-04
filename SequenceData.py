# Import PyTorch dataset utility
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path

# Define the custom dataset class for MIDI token sequences
class MIDITokenDataset(Dataset):
    def __init__(self, token_file_path, token_to_idx, seq_len=100):
        self.seq_len = seq_len  # Set the fixed length for each input sequence
        self.token_file_path = token_file_path  # Path to the tokens.jsonl file
        self.token_to_idx = token_to_idx  # Dictionary mapping tokens to integer indices

        # --- Preload file offsets for lazy access ---
        self.line_offsets = []  # Store byte offset for the beginning of each line
        with open(token_file_path, "r") as f:
            while True:
                pos = f.tell()  # Get current byte offset in file
                line = f.readline()  # Read one line
                if not line:
                    break  # End of file
                self.line_offsets.append(pos)  # Record start position of the line

    def __len__(self):
        return len(self.line_offsets)  # Number of lines (samples) in the dataset

    def __getitem__(self, idx):
        # Open the token file and seek to the line offset
        with open(self.token_file_path, "r") as f:
            f.seek(self.line_offsets[idx])  # Jump to the correct byte offset
            line = f.readline()  # Read the line
            tokens = json.loads(line)  # Parse the JSON array of tokens

        # Convert tokens to integer indices
        int_seq = [self.token_to_idx[t] for t in tokens if t in self.token_to_idx]

        # If sequence is long enough, take a random slice of seq_len tokens and a target
        if len(int_seq) > self.seq_len:
            import random
            start = random.randint(0, len(int_seq) - self.seq_len - 1)  # Random start index
            x = int_seq[start:start + self.seq_len]  # Input sequence
            y = int_seq[start + self.seq_len]        # Target token (next token)
        else:
            # If sequence is too short, pad it with 0s and use dummy target
            x = int_seq + [0] * (self.seq_len - len(int_seq))  # Padded input
            y = 0  # Placeholder target

        # Return as PyTorch tensors
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
