# super-resolution-project

## Objective
This project explores image super-resolution using deep learning models.
The idea is to reconstruct high-resolution (HR) images from low-resolution (LR) inputs.

This README describes the steps to install the project and prepare the data.

---

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/aubinbnf/super-resolution-project.git
cd super-resolution-project
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Data preparation

### Option 1 : Without DVC

#### 1. Download the DIV2K dataset
```bash
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

#### 2. Generate LR (low resolution) images
```bash
python3 src/data/prepare_data.py \
    --data_dir data/raw/DIV2K \
    --output_dir data/processed/DIV2K \
    --scales 2 3 4
```

### Option 2 : With DVC

#### 1. Retrieve datasets from DVC remote
```bash
dvc pull
```