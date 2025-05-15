# 🎵 MIDI Prediction Model (LSTM + LAKH Dataset)

This project trains a character-level LSTM model to predict musical sequences using the [LAKH MIDI Dataset](https://colinraffel.com/projects/lmd/). It converts raw MIDI files into tokenized sequences (note on/off, time shifts, velocities) and learns to generate new music note-by-note.

---

## 🔧 Features

- Preprocessing for large-scale MIDI corpora (LAKH)
- Event-based tokenization (note_on, note_off, time_shift, velocity)
- LSTM-based sequence model
- Auto-saves model weights after training
- Generates new MIDI-style event sequences

---

## 🧠 Tech Stack

- Python 3
- [PrettyMIDI](https://github.com/craffel/pretty-midi)
- NumPy
- PyTorch
- tqdm

---
