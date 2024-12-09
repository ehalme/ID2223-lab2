
# ID2223 Lab 2: Fine-tuning a Pre-trained Language Model using Unsloth Framework

This project demonstrates how to fine-tune a pre-trained language model (`unsloth/Llama-3.2-1B-Instruct`) using lightweight techniques such as LoRA adapters. The fine-tuning process is efficient, requiring updates to only 1–10% of model parameters, and supports various configurations for model optimization. The training pipeline is based on the work by unsloth.

## Authors
- Erik Halme (ehalme@kth.se)
- Oskar Pålhagen (palhagen@kth.se)
- Group 13

## Workflow

1. **Environment Setup**:
   - Google Drive is mounted for saving outputs.
   - Required libraries, such as `unsloth`, are installed.

2. **Pipeline**
   - Base model (Llama-3.2-1B-Instruct) and tokenizer is loaded
   - LoRA adapter are added to the model
   - Training dataset (FineTome-100k) is loaded and converted to the correct format
   - SFTTrainer object is created with config of hyperparameters
   - Models is trained using checkpointing (saved to Google Drive) to allow for multiple sessions of training without losing progress
   - LoRA adapters are saved to HuggingFace at the end of training

## Ways to Improve Model Performance

### (a) Model-Centric Approaches

1. **Hyperparameter Tuning**:
   - Experiment with different values for `max_seq_length`, learning rates, and batch sizes to identify optimal settings.
   - Adjust adapter parameters (e.g., rank and scaling factors for LoRA).

2. **Enhanced Architectures**:
   - Switch to larger pre-trained models like Llama-3.2 with more parameters.
   - Use advanced scaling techniques like dynamic RoPE to improve long-sequence handling.

3. **Regularization**:
   - Apply techniques like dropout or weight decay during fine-tuning to prevent overfitting.

4. **Quantization & Format**:
   - Experiment with mixed precision or 8-bit quantization to improve efficiency while maintaining accuracy.
   - Export model to GGUF format and run using llama.cpp for probably faster CPU inference.

### (b) Data-Centric Approaches

1. **New Data Sources**:
   - Incorporate diverse datasets tailored to your specific use case.
   - Use synthetic data generation tools to augment the dataset.

2. **Data Quality Improvements**:
   - Clean and preprocess the dataset to remove noise and inconsistencies.
   - Balance datasets to avoid biases and improve generalizability.

3. **Task-Specific Data**:
   - Collect real-world user interactions or fine-tuning data for niche applications.

## Getting Started

### Prerequisites

- Python 3.8+
- Libraries: `unsloth`, `torch`, and others specified in the notebook.

### Running the Notebook

1. Clone this repository and open the notebook in Google Colab.
2. Follow the steps to mount Google Drive and install dependencies.
3. Execute cells to load the model and fine-tune it using LoRA.

---

By leveraging both model-centric and data-centric approaches, this project can be extended and refined to achieve better performance tailored to specific applications.
