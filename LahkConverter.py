import os
import pretty_midi
import numpy as np

# Define your event types
TIME_SHIFT_RESOLUTION = 10  # in ms
MAX_SHIFT = 1000            # max shift in ms
NOTE_RANGE = range(21, 109) # piano range

def midi_to_events(pm):
    events = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            # Add note on
            events.append((note.start, f"note_on_{note.pitch}"))
            # Add note off
            events.append((note.end, f"note_off_{note.pitch}"))
    events.sort()  # sort by timestamp

    # Convert to discrete events with time shifts
    token_sequence = []
    last_time = 0.0
    for time, event in events:
        delta = time - last_time
        shift_ms = int(delta * 1000)

        # Break long shifts into multiple steps
        while shift_ms > 0:
            step = min(shift_ms, MAX_SHIFT)
            token_sequence.append(f"time_shift_{step}")
            shift_ms -= step

        token_sequence.append(event)
        last_time = time

    return token_sequence

def build_dataset(midi_dir, limit=1000):
    token_sequences = []
    count = 0

    for root, _, files in os.walk(midi_dir):
        for file in files:
            if not file.endswith('.mid') and not file.endswith('.midi'):
                continue
            try:
                midi_path = os.path.join(root, file)
                pm = pretty_midi.PrettyMIDI(midi_path)
                tokens = midi_to_events(pm)
                token_sequences.append(tokens)
                count += 1
                if count >= limit:
                    return token_sequences
            except Exception as e:
                print(f"Error with {file}: {e}")
                continue

    return token_sequences