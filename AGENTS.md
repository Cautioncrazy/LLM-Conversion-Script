# Gemma 3 1B Abliterated - MediaPipe Integration Guide

This document outlines the technical specifications and deployment instructions for the **Gemma 3 1B Abliterated** model converted into a MediaPipe-compatible `.bin` bundle.

## Model Specifications

- **Base Model**: Gemma 3
- **Parameter Count**: 1 Billion (1B)
- **Status**: Abliterated / Uncensored (safety alignment removed/modified)
- **Quantization**: 4-bit weight-only quantization
- **Target Backend**: GPU/NPU (optimized for Edge environments)
- **Bundle Format**: MediaPipe GenAI Task `.bin` (TFLite based)
- **Output File**: `gemma3_1b_abliterated.bin`

## Deployment Instructions

To deploy this specific `.bin` file into a MediaPipe LLM Inference environment, follow these steps:

### 1. Prerequisites

Ensure you have MediaPipe installed in your deployment environment.

**For Web (JavaScript/TypeScript):**
```bash
npm install @mediapipe/tasks-genai
```

**For Python:**
```bash
pip install mediapipe
```

**For Android/iOS:**
Ensure you have the latest MediaPipe Tasks SDK integrated into your `build.gradle` or `Podfile`.

### 2. Loading the Model in Python

```python
import mediapipe as mp
from mediapipe.tasks.python.genai import llm_inference

# Path to the converted model
model_path = "gemma3_1b_abliterated.bin"

# Initialize the inference options
options = llm_inference.LlmInferenceOptions(
    model_path=model_path,
    max_tokens=1024,
    top_k=40,
    temperature=0.8,
)

# Create the LLM Inference task
llm = llm_inference.LlmInference.create_from_options(options)

# Run inference
response = llm.generate("Explain the theory of relativity.")
print(response)
```

### 3. Loading the Model in Web (JavaScript)

```javascript
import { LlmInference } from '@mediapipe/tasks-genai';

async function runModel() {
  const llmInference = await LlmInference.createFromOptions({
    baseOptions: {
      modelAssetPath: 'gemma3_1b_abliterated.bin'
    },
    maxTokens: 1024,
    topK: 40,
    temperature: 0.8
  });

  const response = await llmInference.generateResponse("What is the meaning of life?");
  console.log(response);
}

runModel();
```

### 4. Important Considerations for the Abliterated Model

* **Content Moderation**: Because this model is "abliterated" (uncensored), it lacks the standard safety guardrails present in the base Gemma model. It may generate content that is unsafe, biased, or inappropriate. If you are deploying this to end-users, you *must* implement your own content filtering and moderation layers on the output.
* **Edge Optimization**: The model utilizes 4-bit weight quantization optimized for GPU backends. Ensure your target deployment environment has WebGL (for Web) or appropriate GPU delegates (for mobile/desktop) enabled to achieve optimal inference speed.
