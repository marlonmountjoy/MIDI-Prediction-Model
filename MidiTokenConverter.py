import os
import sys
import json
import pretty_midi

# Make sure you pass your MIDI folder on the command line:
#   python MidiTokenConverter.py /path/to/lmd_full
if len(sys.argv) < 2:
    print("Usage: python MidiTokenConverter.py <midi_folder>")
    sys.exit(1)

MIDI_DIR    = sys.argv[1]
OUTPUT_FILE = 'tokens.jsonl'
TIME_STEP   = 0.05   # same as before, seconds per time_shift
MAX_SHIFT   = 1000   # ms
NOTE_RANGE  = range(21, 109)

def midi_to_tokens(pm):
    events = []
    for inst in pm.instruments:
        if inst.is_drum: continue
        for note in inst.notes:
            events.append((note.start,     f"note_on_{note.pitch}"))
            events.append((note.end,       f"note_off_{note.pitch}"))
    events.sort()

    seq = []
    last_time = 0.0
    for t, ev in events:
        delta = t - last_time
        ms    = int(delta * 1000)
        while ms > 0:
            step = min(ms, MAX_SHIFT)
            seq.append(f"time_shift_{step}")
            ms   -= step
        seq.append(ev)
        last_time = t
    return seq

with open(OUTPUT_FILE, 'w') as out:
    for root, _, files in os.walk(MIDI_DIR):
        for fname in files:
            if not fname.lower().endswith(('.mid','.midi')):
                continue
            path = os.path.join(root, fname)
            try:
                pm     = pretty_midi.PrettyMIDI(path)
                tokens = midi_to_tokens(pm)
                if tokens:
                    out.write(json.dumps(tokens) + '\n')
            except Exception as e:
                print(f"Warning: skipping {path}: {e}")

print(f"\nDone!  Wrote tokens for all valid MIDI files to {OUTPUT_FILE}")
