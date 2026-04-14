# Gemma 3 1B to MediaPipe Edge Converter

This project provides tools to convert a local Gemma 3 1B model into a 4-bit quantized MediaPipe bundle (`.bin`) optimized for Google AI Edge deployment.

## Usage

1. **Download the Model**: First, ensure you have downloaded the base model files (e.g., using the `huggingface-cli`) into a local directory named `gemma-1b` at the root of this project.

   ```bash
   huggingface-cli download <model-repo-id> --local-dir ./gemma-1b
   ```

2. **Run the Converter**: You can execute the conversion using either the command line or the graphical interface.

   **Command Line Interface:**
   ```bash
   python convert_to_edge.py
   ```
   This will read the model from `./gemma-1b`, apply 4-bit weight quantization, and generate a `gemma3_1b_abliterated.bin` file in the current directory.

   **Graphical Interface (GUI):**
   ```bash
   python gui_converter.py
   ```
   This will launch a modernized desktop window where you can visually select your input and output directories, adjust conversion settings (like Model Type and Backend), and view the progress logs in real-time.

## Known Bugs & Limitations

* **Dependencies**: The script strictly requires `mediapipe` and `tensorflow`. If they are not installed, the script will halt. You can install them via:
  ```bash
  pip install mediapipe tensorflow
  ```
* **Memory Requirements**: Converting a 1B model requires significant RAM. Ensure your system has at least 8-16GB of free memory before starting the conversion.
* **Backend Targeting**: The conversion script is currently hardcoded to optimize for the `gpu` backend using 4-bit weights, suitable for Edge/NPU environments. Changing this requires manual modification of the `convert_to_edge.py` script.
* **Input Directory**: The script strictly expects the model files to be located in a `./gemma-1b` folder relative to where the script is executed.

## Deployment

Please refer to `AGENTS.md` for specific instructions on deploying the resulting `gemma3_1b_abliterated.bin` model into MediaPipe environments, as well as notes regarding its "abliterated" status.
