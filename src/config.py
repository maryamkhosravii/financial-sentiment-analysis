import torch
from pathlib import Path

#Model
MODEL_NAME = "ProsusAI/finbert"
NUM_LABELS = 3
MAX_LENGTH = 64

#Training
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01

#Device
DEVICE = torch.device ("cuda" if torch.cuda.is_available() else "cpu")

#Reproducibility
RANDOM_SEED = 42

#Data
NUM_WORKERS = 2

#Scheduler
WARMUP_RATIO = 0.1

#Paths
PROJECT_ROOT = Path (__file__).parent.parent
MODEL_SAVE_DIR = PROJECT_ROOT/"models"
MODEL_SAVE_NAME = "best_model.pt"
