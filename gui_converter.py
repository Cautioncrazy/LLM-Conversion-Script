import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import sys
import os
import queue

class PrintLogger:
    def __init__(self, message_queue):
        self.message_queue = message_queue

    def write(self, message):
        self.message_queue.put(message)

    def flush(self):
        pass

class ConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gemma 3 1B to MediaPipe Edge Converter")
        self.root.geometry("700x600")

        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Configure fonts and padding for a modernized look
        self.style.configure("TLabel", font=("Segoe UI", 10))
        self.style.configure("TButton", font=("Segoe UI", 10), padding=5)
        self.style.configure("TCombobox", font=("Segoe UI", 10), padding=5)
        self.style.configure("TEntry", font=("Segoe UI", 10), padding=5)
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20 20 20 20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Label(main_frame, text="LLM to MediaPipe Converter", style="Header.TLabel")
        header.pack(pady=(0, 20))

        # Config Frame
        config_frame = ttk.LabelFrame(main_frame, text="Settings", padding="15 15 15 15")
        config_frame.pack(fill=tk.X, pady=(0, 15))

        # Input Directory
        ttk.Label(config_frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_dir_var = tk.StringVar(value=os.path.abspath("./gemma-1b"))
        ttk.Entry(config_frame, textvariable=self.input_dir_var, width=40).grid(row=0, column=1, padx=10, pady=5)
        ttk.Button(config_frame, text="Browse", command=self.browse_input).grid(row=0, column=2, pady=5)

        # Output Directory
        ttk.Label(config_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar(value=os.path.abspath("."))
        ttk.Entry(config_frame, textvariable=self.output_dir_var, width=40).grid(row=1, column=1, padx=10, pady=5)
        ttk.Button(config_frame, text="Browse", command=self.browse_output).grid(row=1, column=2, pady=5)

        # Output Filename
        ttk.Label(config_frame, text="Output Filename:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_filename_var = tk.StringVar(value="gemma3_1b_abliterated.bin")
        ttk.Entry(config_frame, textvariable=self.output_filename_var, width=40).grid(row=2, column=1, padx=10, sticky=tk.W, pady=5)

        # Model Type
        ttk.Label(config_frame, text="Model Type:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.model_type_var = tk.StringVar(value="GEMMA")
        ttk.Combobox(config_frame, textvariable=self.model_type_var, values=["GEMMA", "GEMMA_2B", "PHI_2", "FALCON_1B", "LLAMA"], state="readonly", width=15).grid(row=3, column=1, padx=10, sticky=tk.W, pady=5)

        # Checkpoint Format
        ttk.Label(config_frame, text="Checkpoint Format:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.ckpt_format_var = tk.StringVar(value="safetensors")
        ttk.Combobox(config_frame, textvariable=self.ckpt_format_var, values=["safetensors", "pytorch"], state="readonly", width=15).grid(row=4, column=1, padx=10, sticky=tk.W, pady=5)

        # Backend
        ttk.Label(config_frame, text="Backend:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.backend_var = tk.StringVar(value="gpu")
        ttk.Combobox(config_frame, textvariable=self.backend_var, values=["gpu", "cpu"], state="readonly", width=15).grid(row=5, column=1, padx=10, sticky=tk.W, pady=5)

        # Start Button
        self.start_button = ttk.Button(main_frame, text="Start Conversion", command=self.start_conversion)
        self.start_button.pack(pady=10)

        # Progress Bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', length=400)
        self.progress.pack(pady=10)

        # Console Log
        ttk.Label(main_frame, text="Console Output:", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.console = scrolledtext.ScrolledText(main_frame, height=12, bg="#1e1e1e", fg="#cccccc", font=("Consolas", 9))
        self.console.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.console.configure(state='disabled')

        # Thread-safe logging queue
        self.log_queue = queue.Queue()
        self.logger = PrintLogger(self.log_queue)

        # Redirect stdout and stderr
        sys.stdout = self.logger
        sys.stderr = self.logger

        # Start queue polling
        self.root.after(100, self.process_log_queue)

    def process_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.console.configure(state='normal')
                self.console.insert(tk.END, msg)
                self.console.see(tk.END)
                self.console.configure(state='disabled')
        except queue.Empty:
            pass
        self.root.after(100, self.process_log_queue)

    def browse_input(self):
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_dir_var.set(directory)

    def browse_output(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)

    def log(self, message):
        print(message)

    def start_conversion(self):
        # Disable button and start progress
        self.start_button.config(state=tk.DISABLED)
        self.progress.start(10)
        self.console.configure(state='normal')
        self.console.delete(1.0, tk.END)
        self.console.configure(state='disabled')

        # Run conversion in a separate thread
        thread = threading.Thread(target=self.run_conversion_thread)
        thread.daemon = True
        thread.start()

    def run_conversion_thread(self):
        try:
            self.log("Initializing conversion process...\n")

            try:
                import mediapipe as mp
                import tensorflow as tf
            except ImportError as e:
                self.log("Error: Missing required dependencies.\n")
                self.log(f"Details: {e}\n")
                self.log("Please install the required packages by running:\n")
                self.log("    pip install mediapipe tensorflow\n")
                self.conversion_finished()
                return

            try:
                from mediapipe.tasks.python.genai import converter
            except ImportError as e:
                self.log("Error: Could not import MediaPipe LLM converter tools.\n")
                self.log(f"Details: {e}\n")
                self.log("Please ensure you have a recent version of MediaPipe installed.\n")
                self.conversion_finished()
                return

            input_ckpt = self.input_dir_var.get()
            output_dir = self.output_dir_var.get()
            output_tflite_file = self.output_filename_var.get()

            if not os.path.exists(input_ckpt):
                self.log(f"Error: Input directory '{input_ckpt}' does not exist.\n")
                self.conversion_finished()
                return

            vocab_model_file = os.path.join(input_ckpt, "tokenizer.model")

            self.log("Configuration parameters:\n")
            self.log(f"  Input: {input_ckpt}\n")
            self.log(f"  Output Dir: {output_dir}\n")
            self.log(f"  Filename: {output_tflite_file}\n")
            self.log(f"  Model Type: {self.model_type_var.get()}\n")
            self.log(f"  Format: {self.ckpt_format_var.get()}\n")
            self.log(f"  Backend: {self.backend_var.get()}\n")
            self.log("\nStarting MediaPipe conversion. This may take a while and require significant memory...\n")

            # MediaPipe only supports Gemma models via safetensors checkpoints
            ckpt_format = self.ckpt_format_var.get()
            model_type = self.model_type_var.get()
            if "GEMMA" in model_type and ckpt_format != "safetensors":
                self.log(f"Error: MediaPipe currently only supports 'safetensors' format for Gemma models, not '{ckpt_format}'.\n")
                self.log("Please select 'safetensors' in the Checkpoint Format dropdown.\n")
                self.conversion_finished()
                return

            config = converter.ConversionConfig(
                input_ckpt=input_ckpt,
                ckpt_format=ckpt_format,
                model_type=model_type,
                backend=self.backend_var.get(),
                output_dir=output_dir,
                combine_file_only=False,
                vocab_model_file=vocab_model_file,
                output_tflite_file=os.path.join(output_dir, output_tflite_file),
            )

            converter.convert_checkpoint(config)
            self.log(f"\nSuccessfully converted model to {os.path.join(output_dir, output_tflite_file)}\n")

        except Exception as e:
            self.log(f"\nError during conversion: {e}\n")
        finally:
            self.conversion_finished()

    def conversion_finished(self):
        # Update UI from main thread
        self.root.after(0, self._stop_ui)

    def _stop_ui(self):
        self.progress.stop()
        self.start_button.config(state=tk.NORMAL)
        self.log("Process complete.\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = ConverterGUI(root)
    root.mainloop()
