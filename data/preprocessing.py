# data/preprocessing.py
import os
import cv2
import torchaudio
import torch
import numpy as np
from torchvision import transforms
# import face_alignment # --> You would uncomment and use this

import hyperparameters as hp

# --- Placeholder for Face Alignment ---
# This is complex. In a real scenario, you'd use a library like 'face_alignment'.
# It would detect faces, find landmarks, and warp the image to align the face.
# For now, we'll assume faces are roughly centered or this step happens offline.
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=hp.DEVICE)

def align_face(frame):
    """
    Placeholder for face alignment.
    Should return an aligned face image of a fixed size, or None if no face found.
    """
    # --- Actual Face Alignment Logic Would Go Here ---
    # Example:
    # preds = fa.get_landmarks(frame)
    # if preds is None:
    #     return None
    # aligned_face = warp_and_crop_face(frame, preds[0], scale=1.5, output_size=hp.FRAME_SIZE)
    # return aligned_face
    # --- Placeholder Logic ---
    # Simple center crop as a fallback if alignment isn't implemented/fails
    h, w, _ = frame.shape
    min_dim = min(h, w)
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    cropped = frame[start_h:start_h+min_dim, start_w:start_w+min_dim]
    resized = cv2.resize(cropped, hp.FRAME_SIZE, interpolation=cv2.INTER_AREA)
    return resized


# --- Video Processing ---
def extract_frames(video_path, target_fps=hp.TARGET_FPS):
    """
    Extracts frames from a video file at the target FPS.
    Returns a list of NumPy arrays (frames).
    Requires OpenCV (cv2).
    """
    frames = []
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found {video_path}")
        return frames

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return frames

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(original_fps / target_fps)) if target_fps > 0 else 1
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # --- Optional Face Alignment ---
            # aligned_frame = align_face(frame) # Call alignment here
            # if aligned_frame is not None:
            #     frames.append(aligned_frame)
            # else:
            #     # Handle cases where face is not detected (skip frame, use placeholder?)
            #     print(f"Warning: Face not detected in frame {frame_count} of {video_path}")
            # --- Using simple resize for now ---
            resized_frame = cv2.resize(frame, hp.FRAME_SIZE, interpolation=cv2.INTER_AREA)
            frames.append(resized_frame) # Add frame if face found & aligned

        frame_count += 1

    cap.release()
    return frames # List of HxWxC NumPy arrays

# --- Audio Processing ---
def load_and_resample_audio(audio_path, target_sr=hp.TARGET_SR):
    """Loads and resamples audio using torchaudio."""
    if not os.path.exists(audio_path):
        print(f"Warning: Audio file not found {audio_path}")
        return None, -1

    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        return waveform, target_sr # Returns Tensor [Channels, Time]
    except Exception as e:
        print(f"Error loading or resampling audio {audio_path}: {e}")
        return None, -1

# --- Transforms ---
# Normalization for ResNet models
preprocess_transform = transforms.Compose([
    transforms.ToTensor(), # Converts numpy array (H, W, C) to Tensor (C, H, W) and scales to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])