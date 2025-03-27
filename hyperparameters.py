# hyperparameters.py

# Data paths
AUDIO_PATH = 'path/to/your/audio/data'
VIDEO_PATH = 'path/to/your/video/data'
ANNOTATION_PATH = 'path/to/your/annotation/file'

# Face preprocessing parameters (assuming OpenFace is used externally)
ALIGNED_FACE_SIZE = (224, 224)  # Example size

# ResNet parameters
RESNET_DEPTH = 50  # You can choose between 18, 34, 50, 101, 152
RESNET_PRETRAINED = True  # Use pre-trained weights from ImageNet

# Audio feature parameters
SAMPLE_RATE = 16000
N_MFCC = 40
MFCC_WINDOW_SIZE = 0.025  # in seconds
MFCC_HOP_LENGTH = 0.010   # in seconds

# Emotion prediction module parameters
FEATURE_VECTOR_SIZE = None # This will be determined after defining ResNet and MFCC output sizes
HIDDEN_SIZE = 256
NUM_LAYERS = 2  # For MLP or Transformer
DROPOUT = 0.2

# Transformer parameters (if used)
N_HEAD = 4
TRANSFORMER_DIM_FEEDFORWARD = 512

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4

# Loss function (will be discussed later)
LOSS_TYPE = 'MSELoss'

# Other settings
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42