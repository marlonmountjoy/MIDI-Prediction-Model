import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import subprocess
import os

class MIDIApp:
    def __init__(self, master):
        self.master = master
        master.title("LSTM MIDI Trainer")

        # Dataset selection
        self.dataset_path = tk.StringVar()
        tk.Label(master, text="Training Dataset Folder:").grid(row=0, column=0, sticky="w")
        tk.Entry(master, textvariable=self.dataset_path, width=50).grid(row=1, column=0, padx=5)
        tk.Button(master, text="Browse", command=self.browse_dataset).grid(row=1, column=1)

        # Train button
        tk.Button(master, text="Train Model", command=self.start_training).grid(
            row=2, column=0, pady=10, sticky="w"
        )

        # Generation settings
        tk.Label(master, text="Generate Length (tokens):").grid(row=3, column=0, sticky="w")
        self.gen_length = tk.IntVar(value=1000)
        tk.Entry(master, textvariable=self.gen_length, width=10).grid(row=3, column=1, sticky="w")

        # Generate button
        tk.Button(master, text="Generate Music", command=self.start_generation).grid(
            row=4, column=0, pady=10, sticky="w"
        )

        # Status area
        self.status_text = tk.Text(master, height=12, width=70)
        self.status_text.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

    def browse_dataset(self):
        path = filedialog.askdirectory()
        if path:
            self.dataset_path.set(path)

    def start_training(self):
        threading.Thread(target=self.run_training, daemon=True).start()

    def run_training(self):
        dataset = self.dataset_path.get().strip()
        if not os.path.isdir(dataset):
            messagebox.showerror("Error", "Please select a valid dataset directory.")
            return

        self.update_status("Training started...\n")

        # 1) Clear old files
        for fn in ("tokens.jsonl", "vocab.json"):
            if os.path.exists(fn):
                os.remove(fn)
                self.update_status(f"Removed old {fn}\n")

        try:
            # 2) Convert MIDI → tokens.jsonl
            self.update_status("• Converting MIDI to tokens.jsonl...\n")
            subprocess.run(
                ["python", "MidiTokenConverter.py", dataset],
                check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.update_status("✓ tokens.jsonl created\n")

            # 3) Build vocabulary
            self.update_status("• Building vocab.json...\n")
            subprocess.run(
                ["python", "build_vocab.py"],
                check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.update_status("✓ vocab.json created\n")

            # 4) Train model
            self.update_status("• Training LSTM model...\n")
            subprocess.run(
                ["python", "train_model.py"],
                check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.update_status("✓ model trained and saved\n")

            self.update_status("Training completed successfully!\n")

        except subprocess.CalledProcessError as e:
            # print out any captured stderr for debugging
            err = e.stderr.decode() if e.stderr else str(e)
            self.update_status(f"Training failed:\n{err}\n")

    def start_generation(self):
        threading.Thread(target=self.run_generation, daemon=True).start()

    def run_generation(self):
        length = str(self.gen_length.get())
        self.update_status("Generating music...\n")
        try:
            subprocess.run(
                ["python", "Generate.py", length],
                check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.update_status("Generation completed. Output saved to output/generatedTokens.json\n")
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode() if e.stderr else str(e)
            self.update_status(f"Generation failed:\n{err}\n")

    def update_status(self, message):
        self.status_text.insert(tk.END, message)
        self.status_text.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = MIDIApp(root)
    root.mainloop()
