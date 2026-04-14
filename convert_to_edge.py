import sys
import os
import traceback

def main():
    try:
        import mediapipe as mp
        import tensorflow as tf
    except ImportError as e:
        print("Error: Missing required dependencies.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        print("Please install the required packages by running:", file=sys.stderr)
        print("    pip install mediapipe tensorflow", file=sys.stderr)
        sys.exit(1)

    try:
        from mediapipe.tasks.python.genai import converter
    except ImportError as e:
        print("Error: Could not import MediaPipe LLM converter tools.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        print("Please ensure you have a recent version of MediaPipe installed.", file=sys.stderr)
        sys.exit(1)

    print("Dependencies loaded successfully. Configuring conversion...")

    # Path to the source directory (assumed to contain the model and tokenizer)
    input_ckpt = os.path.abspath("./gemma-1b")
    # Target output directory
    output_dir = os.path.abspath(".")
    output_tflite_file = "gemma3_1b_abliterated.bin"

    # Check if the input directory exists
    if not os.path.exists(input_ckpt):
        print(f"Warning: Input directory '{input_ckpt}' does not exist.", file=sys.stderr)
        print("Please ensure the model files are downloaded via HF CLI before running the conversion.", file=sys.stderr)

    # Path to the tokenizer model file
    vocab_model_file = os.path.join(input_ckpt, "tokenizer.model")

    # Configure the conversion
    # Using 4-bit quantization as requested, optimized for GPU/NPU
    # backend="gpu" implies 4-bit weight quantization for LLMs in MediaPipe

    # MediaPipe only supports Gemma models via safetensors checkpoints
    ckpt_format = "safetensors"
    model_type = "GEMMA_2B"

    config = converter.ConversionConfig(
        input_ckpt=input_ckpt,
        ckpt_format=ckpt_format,
        model_type=model_type,
        backend="gpu",
        output_dir=output_dir,
        combine_file_only=False,
        vocab_model_file=vocab_model_file,
        output_tflite_file=os.path.join(output_dir, output_tflite_file),
    )

    print(f"Converting model from {input_ckpt} to {output_tflite_file}...")

    try:
        # Run the checkpoint conversion
        converter.convert_checkpoint(config)
        print(f"Successfully converted model to {output_tflite_file}")
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        print("--- Detailed Traceback ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("--------------------------", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
