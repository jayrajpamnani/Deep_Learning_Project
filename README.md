# Deep Learning Model Merging Project

This repository contains implementations of two advanced model merging techniques for large language models (LLMs): **TIES (Trim, Elect, and Merge)** and **SLERP (Spherical Linear Interpolation)**. This project was developed as part of the Engineering Applications of Deep Learning (EADL) course.

## Overview

The project demonstrates how to merge multiple fine-tuned language models into a single, more capable model. We merge two Mistral-7B based models:
- **OpenHermes-2.5-Mistral-7B**: A fine-tuned model optimized for helpful, harmless, and honest conversations
- **MythoMist-7b**: A fine-tuned model with enhanced capabilities

Both merging techniques combine these models with the base **Mistral-7B-v0.1** model to create merged models that potentially inherit the strengths of all three.

## Merging Methods

### 1. TIES (Trim, Elect, and Merge)

**File**: `EADL_Model_Merge_TIES.ipynb`

TIES is a robust merging method that:
1. **Trim**: Removes redundant parameters by keeping only the top-k% most significant parameter changes
2. **Elect**: Resolves sign conflicts by electing the majority sign for each parameter
3. **Merge**: Averages the elected parameters to create the final merged model

**Key Features**:
- Handles vocabulary size mismatches automatically
- Optimized for GPU/CPU computation
- Robust error handling for memory constraints

**Parameters**:
- `density`: 0.5 (keeps top 50% of parameter changes)
- `lam`: 1.0 (scaling factor for merged deltas)

### 2. SLERP (Spherical Linear Interpolation)

**File**: `EADL_Model_Merge_SLERP.ipynb`

SLERP performs interpolation along the surface of a hypersphere in the weight space, preserving the magnitude (variance) of weights, which is crucial for maintaining LLM stability.

**Key Features**:
- Preserves weight magnitudes better than linear interpolation
- Automatically falls back to LERP for colinear vectors
- High-precision computation using float32 for interpolation

**Parameters**:
- `merge_ratio`: 0.5 (equal mix of both models)

## Requirements

- Python 3.8+
- PyTorch
- Transformers library
- NumPy
- Google Colab (for running the notebooks as-is)
- Hugging Face account (for accessing Mistral base model)

## Installation

```bash
pip install torch transformers accelerate safetensors huggingface_hub numpy tqdm
```

## Usage

### Running the Notebooks

1. **Open in Google Colab**: The notebooks are designed to run in Google Colab with GPU support (A100 recommended)

2. **Authenticate with Hugging Face**: 
   - You'll need a Hugging Face token to access the Mistral base model
   - Run `notebook_login()` when prompted

3. **Mount Google Drive**: The merged models are saved to your Google Drive

4. **Execute Cells**: Run all cells sequentially

### Customization

You can modify the following parameters in the notebooks:

**TIES Method**:
- `density`: Controls sparsity (0.0 to 1.0)
- `lam`: Scaling factor for merged deltas
- `base_model_name`: Base model to use
- `model_paths`: List of fine-tuned models to merge

**SLERP Method**:
- `merge_ratio`: Interpolation factor (0.0 = 100% Model A, 1.0 = 100% Model B)
- `MODEL_1_URL` and `MODEL_2_URL`: Models to merge

## Project Structure

```
Deep_Learning_Project/
├── EADL_Model_Merge_TIES.ipynb    # TIES merging implementation
├── EADL_Model_Merge_SLERP.ipynb  # SLERP merging implementation
├── EADL_Report.pdf                # Project report
├── Project_Proposal.pdf           # Initial project proposal
└── README.md                       # This file
```

## Results

Both methods successfully create merged models that:
- Maintain model stability and coherence
- Combine capabilities from multiple fine-tuned models
- Handle vocabulary size mismatches automatically
- Can be used for inference with standard transformers pipeline

## Testing

Each notebook includes a test cell that:
- Loads the merged model
- Generates a sample response
- Demonstrates the model's functionality

Example test prompt: *"Explain the concept of quantum entanglement to a 5-year-old."*

## Notes

- The notebooks require significant computational resources (~28GB RAM for loading two 7B models)
- GPU acceleration is recommended for faster merging
- Models are saved in SafeTensors format for efficient storage
- Vocabulary size mismatches are automatically handled by padding

## References

- **TIES Merging**: [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708)
- **SLERP**: Spherical Linear Interpolation for model merging
- **Base Models**: 
  - [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
  - [OpenHermes-2.5-Mistral-7B](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B)
  - [MythoMist-7b](https://huggingface.co/Gryphe/MythoMist-7b)

## Course Information

This project was developed for the **Engineering Applications of Deep Learning** course under Prof. Eugene Vinitsky.

## License

This project is for educational purposes. Please refer to the licenses of the base models and fine-tuned models used in this project.

