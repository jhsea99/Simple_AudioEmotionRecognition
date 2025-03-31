# train.py
import torch
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm # Progress bar

import hyperparameters as hp
from data.dataset import EmotionDataset
from data.dataloader import create_dataloaders
from model.face_feature_extractor import build_face_feature_extractor
from model.audio_feature_extractor import AudioFeatureExtractor
from model.emotion_predictor import EmotionPredictorMLP
from model.loss import get_loss_function, calculate_metrics, concordance_correlation_coefficient

def train_epoch(face_model, audio_model, predictor_model, dataloader, criterion, optimizer, device):
    face_model.eval() # Face model is frozen or just used for feature extraction
    audio_model.train() # Audio model might have trainable components (less likely for simple MFCC)
    predictor_model.train() # Predictor MLP needs training

    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        frames = batch['video_frames'].to(device) # [Batch, N_Frames, C, H, W]
        waveforms = batch['audio_waveform'].to(device) # [Batch, Channels, Time]
        labels = batch['labels'].to(device) # [Batch, 2]
        num_frames_batch = batch['num_frames'] # List or Tensor of frame counts

        # --- Feature Extraction ---
        batch_size = frames.shape[0]
        aggregated_face_features = []

        # Process face frames per item in batch (handle varying frame counts)
        # This is less efficient than batching frames directly if possible
        # Consider restructuring if performance is critical (e.g., padding frames)
        with torch.no_grad(): # No gradients needed for frozen face extractor
             for i in range(batch_size):
                 n_frames = num_frames_batch[i].item()
                 if n_frames > 0:
                     # Pass frames through ResNet [N_Frames, C, H, W] -> [N_Frames, FaceFeatDim]
                     item_frames = frames[i, :n_frames] # Select actual frames for this item
                     frame_features = face_model(item_frames)
                     # Aggregate features (e.g., mean over frames) -> [1, FaceFeatDim]
                     aggregated_feature = torch.mean(frame_features, dim=0, keepdim=True)
                 else:
                     # Handle items with no valid frames (e.g., zero features)
                     aggregated_feature = torch.zeros(1, hp.FACE_FEATURE_DIM, device=device)
                 aggregated_face_features.append(aggregated_feature)

             # Combine aggregated features into a batch tensor -> [Batch, FaceFeatDim]
             face_features_batch = torch.cat(aggregated_face_features, dim=0)

        # Extract audio features -> [Batch, AudioFeatDim]
        # Note: AudioFeatureExtractor handles aggregation internally
        audio_features_batch = audio_model(waveforms)

        # --- Prediction ---
        optimizer.zero_grad()

        # Forward pass through predictor (uses both modalities for training)
        # `use_face=True` is implicit in the EmotionPredictorMPL's logic
        # when face_features are provided.
        predictions = predictor_model(face_features_batch, audio_features_batch, use_face=True)

        # --- Loss & Backpropagation ---
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size # Accumulate loss weighted by batch size

        all_preds.append(predictions.detach())
        all_labels.append(labels.detach())

        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(dataloader.dataset) # Average loss over all samples
    # Concatenate all predictions and labels for epoch metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    epoch_metrics = calculate_metrics(all_preds, all_labels) # Use torch tensors

    return epoch_loss, epoch_metrics


def validate_epoch(face_model, audio_model, predictor_model, dataloader, criterion, device):
    face_model.eval()
    audio_model.eval()
    predictor_model.eval() # Set predictor to evaluation mode

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # No need to track gradients during validation
        pbar = tqdm(dataloader, desc="Validating")
        for batch in pbar:
            frames = batch['video_frames'].to(device)
            waveforms = batch['audio_waveform'].to(device)
            labels = batch['labels'].to(device)
            num_frames_batch = batch['num_frames']

            # --- Feature Extraction (same as training) ---
            batch_size = frames.shape[0]
            aggregated_face_features = []
            for i in range(batch_size):
                 n_frames = num_frames_batch[i].item()
                 if n_frames > 0:
                     item_frames = frames[i, :n_frames]
                     frame_features = face_model(item_frames)
                     aggregated_feature = torch.mean(frame_features, dim=0, keepdim=True)
                 else:
                     aggregated_feature = torch.zeros(1, hp.FACE_FEATURE_DIM, device=device)
                 aggregated_face_features.append(aggregated_feature)
            face_features_batch = torch.cat(aggregated_face_features, dim=0)

            audio_features_batch = audio_model(waveforms)

            # --- Prediction (using both modalities for validation) ---
            predictions = predictor_model(face_features_batch, audio_features_batch, use_face=True)

            # --- Loss Calculation ---
            loss = criterion(predictions, labels)
            running_loss += loss.item() * batch_size

            all_preds.append(predictions)
            all_labels.append(labels)
            pbar.set_postfix({'val_loss': loss.item()})

    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    epoch_metrics = calculate_metrics(all_preds, all_labels) # Use torch tensors

    return epoch_loss, epoch_metrics

def test_audio_only(audio_model, predictor_model, dataloader, device):
    """ Function to demonstrate audio-only testing """
    audio_model.eval()
    predictor_model.eval()

    all_preds = []
    all_labels = [] # Optional: if your test set has labels

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing (Audio Only)")
        for batch in pbar:
            # Only load audio and labels (if available)
            waveforms = batch['audio_waveform'].to(device)
            labels = batch['labels'].to(device) # If test labels exist

            # Extract audio features
            audio_features_batch = audio_model(waveforms)

            # --- Prediction (AUDIO ONLY) ---
            # Pass None or let the model create dummy features internally
            predictions = predictor_model(face_features=None, # Pass None
                                          audio_features=audio_features_batch,
                                          use_face=False) # Explicitly tell model not to use face

            all_preds.append(predictions)
            if labels is not None:
                 all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0)
    if all_labels:
        all_labels = torch.cat(all_labels, dim=0)
        test_metrics = calculate_metrics(all_preds, all_labels)
        print("\n--- Audio-Only Test Metrics ---")
        for name, value in test_metrics.items():
            print(f"{name}: {value:.4f}")
        print("-----------------------------")
        return test_metrics
    else:
        print("\n--- Audio-Only Test Predictions Generated ---")
        # You might save 'all_preds' here
        return None # No metrics if no labels


def main():
    # --- Setup ---
    device = torch.device(hp.DEVICE)
    os.makedirs(hp.MODEL_SAVE_DIR, exist_ok=True)

    # --- Models ---
    print("Building models...")
    face_model = build_face_feature_extractor().to(device)
    audio_model = AudioFeatureExtractor().to(device) # MFCC calculation often faster on CPU, but keep consistent
    predictor_model = EmotionPredictorMLP().to(device)

    # --- Data ---
    print("Loading dataset...")
    # Ensure paths in hyperparameters.py are correct
    dataset = EmotionDataset(label_file=hp.LABEL_FILE,
                             video_dir=hp.VIDEO_DIR,
                             audio_dir=hp.AUDIO_DIR)
    train_loader, val_loader = create_dataloaders(dataset)

    # --- Loss & Optimizer ---
    criterion = get_loss_function(hp.LOSS_FUNCTION)
    # Only optimize predictor parameters if face model is frozen
    params_to_optimize = list(predictor_model.parameters())
    if not hp.FREEZE_FACE_EXTRACTOR:
         print("Including face model parameters in optimizer.")
         params_to_optimize += list(face_model.parameters())
    # Add audio model params if it has any trainable ones (unlikely for basic MFCC)
    # params_to_optimize += list(audio_model.parameters())

    if hp.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(params_to_optimize, lr=hp.LEARNING_RATE)
    elif hp.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(params_to_optimize, lr=hp.LEARNING_RATE, momentum=0.9)
    else:
         raise ValueError(f"Unsupported optimizer: {hp.OPTIMIZER}")

    # --- Training Loop ---
    best_val_metric = -float('inf') # Initialize for maximization (e.g., CCC)
    # If using MSE/MAE loss for optimization, you might want to minimize that
    # best_val_metric = float('inf') # Initialize for minimization (e.g., MSE Loss)
    metric_to_monitor = 'CCC_AVG' # Choose metric to decide 'best' model (e.g., CCC_AVG or MSE_AVG)

    print(f"Starting training for {hp.EPOCHS} epochs...")
    for epoch in range(hp.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{hp.EPOCHS} ---")

        # Training
        train_loss, train_metrics = train_epoch(face_model, audio_model, predictor_model,
                                                train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")
        for name, value in train_metrics.items(): print(f"  Train {name}: {value:.4f}")

        # Validation
        val_loss, val_metrics = validate_epoch(face_model, audio_model, predictor_model,
                                               val_loader, criterion, device)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")
        for name, value in val_metrics.items(): print(f"  Val {name}: {value:.4f}")

        # --- Save Checkpoint & Best Model ---
        current_val_metric = val_metrics.get(metric_to_monitor, None)

        if current_val_metric is None:
            print(f"Warning: Metric '{metric_to_monitor}' not found in validation metrics.")
            continue # Skip saving logic if metric is missing


        # Example: Saving based on maximizing CCC_AVG
        is_best = current_val_metric > best_val_metric

        # Example: Saving based on minimizing MSE_AVG
        # is_best = current_val_metric < best_val_metric # if metric_to_monitor was 'MSE_AVG'


        if is_best:
            print(f"Validation {metric_to_monitor} improved ({best_val_metric:.4f} --> {current_val_metric:.4f}). Saving best model...")
            best_val_metric = current_val_metric
            # Save only the predictor model state, as others might be fixed or loaded separately
            torch.save({
                'epoch': epoch + 1,
                'predictor_state_dict': predictor_model.state_dict(),
                # Include face/audio model state if they are fine-tuned
                 'face_model_state_dict': face_model.state_dict() if not hp.FREEZE_FACE_EXTRACTOR else None,
                 'audio_model_state_dict': audio_model.state_dict(), # Save even if not trained
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_metric': best_val_metric,
                'val_loss': val_loss,
                'metrics': val_metrics
            }, os.path.join(hp.MODEL_SAVE_DIR, hp.BEST_MODEL_FILENAME))

        # Save a checkpoint of the latest epoch (optional)
        torch.save({
            'epoch': epoch + 1,
            'predictor_state_dict': predictor_model.state_dict(),
            'face_model_state_dict': face_model.state_dict() if not hp.FREEZE_FACE_EXTRACTOR else None,
            'audio_model_state_dict': audio_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'train_loss': train_loss,
            'train_metrics': train_metrics
        }, os.path.join(hp.MODEL_SAVE_DIR, hp.CHECKPOINT_FILENAME))


    print("\n--- Training Finished ---")

    # --- Load Best Model and Test (Audio Only) ---
    print(f"Loading best model from: {os.path.join(hp.MODEL_SAVE_DIR, hp.BEST_MODEL_FILENAME)}")
    if os.path.exists(os.path.join(hp.MODEL_SAVE_DIR, hp.BEST_MODEL_FILENAME)):
        checkpoint = torch.load(os.path.join(hp.MODEL_SAVE_DIR, hp.BEST_MODEL_FILENAME), map_location=device)

        # Create new instances or use existing ones before loading state_dict
        # Best practice: create new instances to ensure clean state
        face_model_best = build_face_feature_extractor().to(device)
        audio_model_best = AudioFeatureExtractor().to(device)
        predictor_model_best = EmotionPredictorMLP().to(device)

        if checkpoint.get('face_model_state_dict') is not None:
             face_model_best.load_state_dict(checkpoint['face_model_state_dict'])
        audio_model_best.load_state_dict(checkpoint['audio_model_state_dict'])
        predictor_model_best.load_state_dict(checkpoint['predictor_state_dict'])

        print(f"Best model loaded from epoch {checkpoint['epoch']} with Val {metric_to_monitor}: {checkpoint['best_val_metric']:.4f}")

        # Use the validation loader for testing demonstration, or create a separate test loader
        test_audio_only(audio_model_best, predictor_model_best, val_loader, device)

    else:
        print("Best model file not found. Skipping audio-only test.")


if __name__ == '__main__':
    main()