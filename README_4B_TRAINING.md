# Qwen3-4B Production Training on Google Colab

This repository is prepared to fine-tune `Qwen/Qwen3-4B` for the Coal Mine Governance application using 4-bit QLoRA. 

To bypass local hardware constraints (like a 4GB VRAM limit), this pipeline is designed to be executed in a **Google Colab** environment equipped with a T4 GPU (16GB VRAM) or better (A100/L4).

## Step-by-Step Google Colab Execution Guide

### 1. Open Google Colab & Setup Hardware
1. Go to [Google Colab](https://colab.research.google.com/).
2. Create a **New Notebook**.
3. Go to **Runtime > Change runtime type**.
4. Select **T4 GPU** (or better if you have Colab Pro).
5. Click **Save**.

### 2. Verify GPU Allocation
Create a new cell and run the following command to verify you have a 16GB+ GPU assigned:
```python
!nvidia-smi
```

### 3. Clone Repository & Install Dependencies
Since you need the `coal-gov-ai` repository files, upload this repository folder to your Google Drive, OR clone it if it is hosted on GitHub.

*(Assuming you uploaded the folder `coal-gov-ai` to your Google Drive root):*
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to the repository
%cd /content/drive/MyDrive/coal-gov-ai

# Install the necessary HuggingFace libraries
!pip install -r requirements-gpu.txt
```
*(Colab already has PyTorch + CUDA pre-installed, so the requirements file skips it to save time).*

### 4. Verify the V3 Dataset
The repository should already contain `data/v3_train` and `data/v3_validation`. If they are missing or you want to regenerate them freshly:
```python
!python generate_v3_dataset.py
```

### 5. Execute 4B Training (The Heavy Lifting)
Run the training script. The script uses Gradient Checkpointing and 4-bit Quantization to ensure the 4B model fits inside the 16GB T4 GPU.
```python
!python train_4b.py
```
* **Duration:** This may take 1-3 hours depending on the GPU and the 2500 V3 examples.
* **Outputs:** The adapter will be automatically saved to `./models/coal-gov-4b` within your Google Drive folder, making it persistent even after the Colab session closes!

### 6. Evaluate the Fine-Tuned 4B Model
Run the strict evaluation script against the held-out test set:
```python
!python evaluate_4b.py
```
* This script will output JSON Schema validity, Hallucination checks, and exact semantic accuracy metrics (Event Type, Category, Severity). 
* It also prints a Confusion Matrix to help you identify if the model still struggles with specific hazard distinctions.

## 7. Next Steps: Deploying Back to the Application

Once training is complete, the `coal-gov-4b` folder will reside in your Google Drive. 
1. Download the `./models/coal-gov-4b` folder back to your local machine (or your production deployment server).
2. Start the Inference Server pointing to the new model:
   ```bash
   export BASE_MODEL="Qwen/Qwen3-4B"
   export ADAPTER_DIR="./models/coal-gov-4b"
   export MODEL_ALIAS="coal-gov-4b"
   
   python serve_model.py
   ```
3. Update your main Coal Mine Application's `.env` to point to the server using the alias `coal-gov-4b`. No other application rewrites are necessary.
