import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import subprocess
import os

# Main application class
class MIDIApp:
    def __init__(self, master):
        self.master = master
        master.title("LSTM MIDI Trainer")

        # Dataset selection
        self.dataset_path = tk.StringVar()
        tk.Label(master, text="Training Dataset Folder:").grid(row=0, column=0, sticky="w")
        tk.Entry(master, textvariable=self.dataset_path, width=50).grid(row=1, column=0, padx=5)
        tk.Button(master, text="Browse", command=self.browse_dataset).grid(row=1, column=1)

        # Training Button
        tk.Button(master, text="Train Model", command=self.start_training).grid(row=2, column=0, pady=10, sticky="w")

        # Generation settings
        tk.Label(master, text="Generate Length (tokens):").grid(row=3, column=0, sticky="w")
        self.gen_length = tk.IntVar(value=1000)
        tk.Entry(master, textvariable=self.gen_length, width=10).grid(row=3, column=1, sticky="w")

        # Generate Button
        tk.Button(master, text="Generate Music", command=self.start_generation).grid(row=4, column=0, pady=10, sticky="w")

        # Status area
        self.status_text = tk.Text(master, height=10, width=70)
        self.status_text.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

    def browse_dataset(self):
        path = filedialog.askdirectory()
        if path:
            self.dataset_path.set(path)

    def start_training(self):
        threading.Thread(target=self.run_training).start()

    def run_training(self):
        dataset = self.dataset_path.get()
        if not os.path.isdir(dataset):
            messagebox.showerror("Error", "Please select a valid dataset directory.")
            return
        self.update_status("Training started...\n")
        try:
            subprocess.run(["python", "train_model.py", dataset], check=True)
            self.update_status("Training complete!\n")
        except subprocess.CalledProcessError as e:
            self.update_status(f"Training failed: {e}\n")

    def start_generation(self):
        threading.Thread(target=self.run_generation).start()

    def run_generation(self):
        length = str(self.gen_length.get())
        self.update_status("Generating music...\n")
        try:
            subprocess.run(["python", "Generate.py", length], check=True)
            self.update_status("Generation complete! Output saved.\n")
        except subprocess.CalledProcessError as e:
            self.update_status(f"Generation failed: {e}\n")

    def update_status(self, message):
        self.status_text.insert(tk.END, message)
        self.status_text.see(tk.END)

# Launch GUI
root = tk.Tk()
app = MIDIApp(root)
root.mainloop()