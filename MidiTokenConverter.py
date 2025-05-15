import pretty_midi
import json

# --- Settings ---
INPUT_FILE = "generatedTokens.json"         # JSON file containing generated tokens
OUTPUT_MIDI = "generated_output.mid"         # Output filename for the MIDI file
TIME_SHIFT_RESOLUTION = 3 / 1000            # Time shift step size: 10 ms converted to seconds

# --- Function to convert tokens to a MIDI file ---
def tokens_to_midi(tokens, output_path):
    midi = pretty_midi.PrettyMIDI(initial_tempo=160.0)                      # Create a new MIDI object
    instrument = pretty_midi.Instrument(program=0)       # Use acoustic grand piano (program 0)

    time = 0.0                                            # Current absolute time in seconds
    note_ons = {}                                         # Track note_on events by pitch

    for token in tokens:
        if token.startswith("note_on_"):
            pitch = int(token.split("_")[-1])            # Extract MIDI pitch number
            note_ons[pitch] = time                       # Save the start time for this pitch

        elif token.startswith("note_off_"):
            pitch = int(token.split("_")[-1])            # Extract pitch again
            if pitch in note_ons:                        # Only add if there was a corresponding note_on
                start = note_ons.pop(pitch)              # Get the start time
                end = time                               # End is now
                note = pretty_midi.Note(
                    velocity=100, pitch=pitch, start=start, end=end
                )
                instrument.notes.append(note)            # Add the note to the instrument

        elif token.startswith("time_shift_"):
            shift_ms = int(token.split("_")[-1])         # Extract time shift in ms
            time += shift_ms * TIME_SHIFT_RESOLUTION     # Advance current time

    midi.instruments.append(instrument)                  # Add the instrument to the MIDI track
    midi.write(output_path)                              # Write the MIDI file to disk
    print(f"\u2705 MIDI file written to {output_path}")

# --- Load generated tokens from a JSON file ---
with open(INPUT_FILE, "r") as f:
    tokens = json.load(f)

# --- Convert token list to MIDI and save it ---
tokens_to_midi(tokens, OUTPUT_MIDI)
