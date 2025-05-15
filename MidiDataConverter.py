import os
import json
import pretty_midi
import numpy as np

# --- Configurable settings ---

MIDI_PATH = 'data'         # Folder containing MIDI files
OUTPUT_FILE = 'tokens.jsonl'  # Output file for token sequences (one per line)
MAX_FILES = 100            # Max number of MIDI files to process (adjustable)

# Token settings
TIME_SHIFT_RESOLUTION = 10  # Time resolution for shifts in milliseconds
MAX_SHIFT = 1000            # Max time shift per token (split longer gaps)
NOTE_RANGE = range(21, 109) # Restrict to piano note range (A0 to C8)

# --- Function to convert a PrettyMIDI object to a list of tokens ---
def midi_to_events(pm):
    events = []

    # Loop through all instruments, skipping drums
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        # Add note-on and note-off events as (timestamp, event) pairs
        for note in instrument.notes:
            events.append((note.start, f"note_on_{note.pitch}"))
            events.append((note.end, f"note_off_{note.pitch}"))

    # Sort events in order of their timestamps
    events.sort()

    token_sequence = []
    last_time = 0.0

    # Iterate through the events and insert time shifts
    for time, event in events:
        delta = time - last_time
        shift_ms = int(delta * 1000)  # Convert seconds to milliseconds

        # Break long time gaps into multiple tokens
        while shift_ms > 0:
            step = min(shift_ms, MAX_SHIFT)
            token_sequence.append(f"time_shift_{step}")
            shift_ms -= step

        # Append the actual event (note_on or note_off)
        token_sequence.append(event)
        last_time = time

    return token_sequence

# --- Function to process an entire directory of MIDI files ---
def build_dataset(midi_dir, limit=1000):
    token_sequences = []
    count = 0

    # Walk through all subdirectories and files
    for root, _, files in os.walk(midi_dir):
        for file in files:
            # Only process .mid or .midi files
            if not file.endswith('.mid') and not file.endswith('.midi'):
                continue
            try:
                midi_path = os.path.join(root, file)
                # Load the MIDI file
                pm = pretty_midi.PrettyMIDI(midi_path)

                # Convert it to a sequence of tokens
                tokens = midi_to_events(pm)

                # Skip tiny/empty sequences
                if len(tokens) > 10:
                    token_sequences.append(tokens)
                    count += 1

                # Stop when the file limit is reached
                if count >= limit:
                    return token_sequences
            except Exception as e:
                # Skip corrupted/unreadable MIDI files
                print(f"Error with {file}: {e}")
                continue

    return token_sequences

# --- Main script execution ---
if __name__ == "__main__":
    print(f"\U0001F4C1 Scanning '{MIDI_PATH}' for MIDI files...")

    # Process MIDI files and build token dataset
    dataset = build_dataset(MIDI_PATH, limit=MAX_FILES)

    print(f"\u2705 Processed {len(dataset)} MIDI files.")
    print(f"\U0001F4C4 Saving token sequences to {OUTPUT_FILE}...")

    # Write token sequences to JSONL file (one sequence per line)
    with open(OUTPUT_FILE, "w") as f:
        for seq in dataset:
            json.dump(seq, f)
            f.write('\n')

    # Print a small sample of the first token sequence
    print(f"\u2705 Done. Sample output:\n{dataset[0][:20]}")
