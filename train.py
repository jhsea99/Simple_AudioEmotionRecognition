# train.py (modified to include learning rate scheduling and gradient clipping)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data.dataset import FaceAudioEmotionDataset
from model.face_feature_extractor import FaceFeatureExtractor
from model.audio_feature_extractor import AudioFeatureExtractor
from model.emotion_predictor import EmotionPredictor # Or EmotionPredictorWithTransformer
from hyperparameters import (
    AUDIO_PATH,
    VIDEO_PATH,
    ANNOTATION_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    DEVICE,
    LOSS_TYPE,
    FEATURE_VECTOR_SIZE, # Make sure this is set correctly
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    N_MFCC
)
import numpy as np
import os
from torch.optim.lr_scheduler import StepLR

# Set FEATURE_VECTOR_SIZE based on ResNet output and MFCC mean
RESNET_OUTPUT_SIZE = 2048 # Assuming ResNet-50
FEATURE_VECTOR_SIZE = RESNET_OUTPUT_SIZE + N_MFCC

# Define the directory to save the models
SAVE_DIR = 'saved_models'
os.makedirs(SAVE_DIR, exist_ok=True)

# Instantiate the full dataset
full_dataset = FaceAudioEmotionDataset(AUDIO_PATH, VIDEO_PATH, ANNOTATION_PATH)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(full_dataset)) # Example: 80% for training
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create the data loaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Instantiate the models
face_extractor = FaceFeatureExtractor().to(DEVICE)
audio_extractor = AudioFeatureExtractor()
emotion_predictor = EmotionPredictor(FEATURE_VECTOR_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
# If using Transformer:
# emotion_predictor = EmotionPredictorWithTransformer(FEATURE_VECTOR_SIZE, HIDDEN_SIZE, NUM_LAYERS, N_HEAD, DROPOUT, TRANSFORMER_DIM_FEEDFORWARD).to(DEVICE)

# Define the loss function
if LOSS_TYPE == 'MSELoss':
    criterion = nn.MSELoss()
elif LOSS_TYPE == 'L1Loss':
    criterion = nn.L1Loss()
else:
    raise ValueError(f"Unsupported loss type: {LOSS_TYPE}")
criterion = criterion.to(DEVICE)

# Define the optimizer
optimizer = optim.Adam(emotion_predictor.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Define the learning rate scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.1) # Reduce LR by a factor of 0.1 every 10 epochs

def calculate_metrics(predictions, targets):
    """Calculates Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)."""
    mae = nn.L1Loss()(predictions, targets)
    rmse = torch.sqrt(nn.MSELoss()(predictions, targets))
    return mae.item(), rmse.item()

def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_val_loss = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in data_loader:
            audio_waveforms, face_images, targets = batch
            face_images = face_images.to(device)
            targets = targets.to(device)

            # Extract face features
            face_features = face_extractor(face_images)

            # Extract audio features
            audio_features_list = []
            for waveform in audio_waveforms:
                mfcc_output = audio_extractor.extract_features(waveform.unsqueeze(0))
                mean_mfcc = torch.mean(mfcc_output, dim=2).squeeze(0).to(device)
                audio_features_list.append(mean_mfcc)
            audio_features = torch.stack(audio_features_list).to(device)

            # Concatenate features
            combined_features = torch.cat((face_features, audio_features), dim=1)

            # Predict emotions
            outputs = model(combined_features)

            # Calculate loss
            loss = loss_fn(outputs, targets)
            total_val_loss += loss.item()

            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_val_loss = total_val_loss / len(data_loader)
    all_predictions = torch.tensor(np.array(all_predictions))
    all_targets = torch.tensor(np.array(all_targets))
    valence_mae, valence_rmse = calculate_metrics(all_predictions[:, 0], all_targets[:, 0])
    arousal_mae, arousal_rmse = calculate_metrics(all_predictions[:, 1], all_targets[:, 1])

    return avg_val_loss, valence_mae, valence_rmse, arousal_mae, arousal_rmse

# Training loop
for epoch in range(NUM_EPOCHS):
    emotion_predictor.train()
    total_train_loss = 0
    for batch_idx, batch in enumerate(train_dataloader):
        audio_waveforms, face_images, targets = batch
        face_images = face_images.to(DEVICE)
        targets = targets.to(DEVICE)

        # Extract face features
        face_features = face_extractor(face_images)

        # Extract audio features
        audio_features_list = []
        for waveform in audio_waveforms:
            mfcc_output = audio_extractor.extract_features(waveform.unsqueeze(0))
            mean_mfcc = torch.mean(mfcc_output, dim=2).squeeze(0).to(DEVICE)
            audio_features_list.append(mean_mfcc)
        audio_features = torch.stack(audio_features_list).to(DEVICE)

        # Concatenate features
        combined_features = torch.cat((face_features, audio_features), dim=1)

        # Predict emotions
        outputs = emotion_predictor(combined_features)

        # Calculate loss
        loss = criterion(outputs, targets)
        total_train_loss += loss.item()

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        # Implement gradient clipping
        torch.nn.utils.clip_grad_norm_(emotion_predictor.parameters(), max_norm=1.0)
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_dataloader)}], Training Loss: {loss.item():.4f}')

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Average Training Loss: {avg_train_loss:.4f}')

    # Validation step and evaluation
    avg_val_loss, valence_mae, valence_rmse, arousal_mae, arousal_rmse = evaluate(emotion_predictor, val_dataloader, criterion, DEVICE)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Average Validation Loss: {avg_val_loss:.4f}')
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Valence MAE: {valence_mae:.4f}, Valence RMSE: {valence_rmse:.4f}')
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Arousal MAE: {arousal_mae:.4f}, Arousal RMSE: {arousal_rmse:.4f}')

    # Save the model after each epoch
    model_save_path = os.path.join(SAVE_DIR, f'emotion_predictor_epoch_{epoch+1}.pth')
    torch.save(emotion_predictor.state_dict(), model_save_path)
    print(f'Saved model checkpoint to: {model_save_path}')

    # Update the learning rate
    scheduler.step()
    print(f'Learning rate updated to: {optimizer.param_groups[0]["lr"]}')

print("Training finished!")

# --- Function to load the model ---
def load_model(model_path, input_size, hidden_size, num_layers, dropout, device):
    model = EmotionPredictor(input_size, hidden_size, num_layers, dropout) # Or EmotionPredictorWithTransformer
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode for inference
    return model

# if __name__ == '__main__':
    # Example of loading a saved model (assuming you have a saved model file)
    # Replace 'path/to/your/saved_model.pth' with the actual path
    # loaded_model_path = 'saved_models/emotion_predictor_epoch_50.pth'
    # if os.path.exists(loaded_model_path):
    #     loaded_emotion_predictor = load_model(
    #         loaded_model_path,
    #         FEATURE_VECTOR_SIZE,
    #         HIDDEN_SIZE,
    #         NUM_LAYERS,
    #         DROPOUT,
    #         DEVICE
    #     )
    #     print("Loaded model successfully!")