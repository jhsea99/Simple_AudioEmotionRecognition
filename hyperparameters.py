# hyperparameters.py
import torch

# --- Data ---
BASE_DATA_PATH = './dataset/example/' # Path to your main data directory
LABEL_FILE = './dataset/example/labels.csv' # Path to your CSV file with labels (e.g., video_id, valence, arousal)
VIDEO_DIR = f'{BASE_DATA_PATH}/videos/' # Assumes videos are here
AUDIO_DIR = f'{BASE_DATA_PATH}/audio/' # Assumes pre-extracted audio is here (or extracted on the fly)
PREPROCESSED_DIR = f'{BASE_DATA_PATH}/preprocessed/' # Optional: for storing processed data

# --- Preprocessing ---
TARGET_FPS = 25
TARGET_SR = 16000 # Audio sample rate
FRAME_SIZE = (224, 224) # Input size for ResNet
FACE_ALIGNMENT_METHOD = 'face_alignment' # Or 'dlib', 'mediapipe', etc. - Placeholder

# --- Audio Features ---
N_MFCC = 40 # Number of MFCCs to extract
MFCC_HOP_LENGTH = int(TARGET_SR * 0.010) # 10ms hop size
MFCC_WIN_LENGTH = int(TARGET_SR * 0.025) # 25ms window size
FEATURE_AGGREGATION = 'mean' # How to aggregate features over time ('mean', 'max', etc.)

# --- Model ---
FACE_EXTRACTOR_MODEL = 'resnet18' # 'resnet18' or 'resnet50'
FREEZE_FACE_EXTRACTOR = True # Freeze weights of pre-trained ResNet
AUDIO_FEATURE_DIM = N_MFCC # Dimension of audio features after aggregation
# ResNet output dims (before final fc): resnet18=512, resnet50=2048
RESNET_OUTPUT_DIMS = {'resnet18': 512, 'resnet50': 2048}
FACE_FEATURE_DIM = RESNET_OUTPUT_DIMS[FACE_EXTRACTOR_MODEL]

# MLP predictor config
MLP_HIDDEN_LAYERS = [512, 256] # Example hidden layer sizes
OUTPUT_DIM = 2 # Valence, Arousal

# --- Training ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
OPTIMIZER = 'Adam' # 'Adam', 'SGD', etc.
LOSS_FUNCTION = 'MSE' # 'MSE', 'MAE', 'CCC' (CCC needs custom implementation)
VALIDATION_SPLIT = 0.15 # Fraction of data for validation
NUM_WORKERS = 4 # For DataLoader

# --- Evaluation & Saving ---
EVALUATION_METRICS = ['MSE', 'MAE', 'CCC'] # Metrics to compute
MODEL_SAVE_DIR = './saved_models/'
BEST_MODEL_FILENAME = f'best_emotion_model_{FACE_EXTRACTOR_MODEL}.pth'
CHECKPOINT_FILENAME = f'checkpoint_{FACE_EXTRACTOR_MODEL}.pth'

print(f"--- Hyperparameters ---")
print(f"Device: {DEVICE}")
print(f"Face Model: {FACE_EXTRACTOR_MODEL}")
print(f"Audio Features: {N_MFCC} MFCCs")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Loss Function: {LOSS_FUNCTION}")
print(f"-----------------------")