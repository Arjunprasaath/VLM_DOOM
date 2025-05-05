# Finetuning Vision Language Model to play Doom
This project explores the capabilities of VLM to play games (DOOM). To achieve this custom PPOTrainer class was built since transformer's TRL library does not support finetuning VLM using Reinforcement Learning.

## Installation

Pip install teh requirements file

```bash
pip install -r requirements.txt
```
Also VizDoom is required so git clone if not available already
```bash
git clone https://github.com/Farama-Foundation/ViZDoom.git
```

## Usage

1. Install requirements.
2. Install wandb if posted.
3. Run `doom_vlm_train.py` for training Qwen2.5-VL-3B-Instruct on the Doom env (keep render = False)
4. Run inference on the trained model using `inference.ipynb`
