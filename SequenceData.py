import torch
from torch.utils.data import Dataset
import json
from pathlib import Path

class MIDITokenDataset(Dataset):
    def __init__(self, token_file_path, token_to_idx, seq_len=100):
        self.seq_len = seq_len
        self.samples = []

        # Load and convert each line of tokens into index sequences
        with open(token_file_path, "r") as f:
            for line in f:
                tokens = json.loads(line)
                int_seq = [token_to_idx[t] for t in tokens if t in token_to_idx]

                # Slice into fixed-length windows
                for i in range(0, len(int_seq) - seq_len):
                    x = int_seq[i:i + seq_len]
                    y = int_seq[i + seq_len]  # next token
                    self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
