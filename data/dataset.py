# data/dataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from data import preprocessing
import hyperparameters as hp

class EmotionDataset(Dataset):
    def __init__(self, label_file=hp.LABEL_FILE, video_dir=hp.VIDEO_DIR,
                 audio_dir=hp.AUDIO_DIR, frame_transform=preprocessing.preprocess_transform):
        """
        Args:
            label_file (string): Path to the csv file with annotations.
                                 Expected columns: 'video_id', 'valence', 'arousal'.
            video_dir (string): Directory with all the video files.
            audio_dir (string): Directory with all the audio files.
                                (Assumes audio filenames match video_ids).
            frame_transform (callable, optional): Optional transform to be applied on frames.
        """
        print(f"Loading labels from: {label_file}")
        self.labels_df = pd.read_csv(label_file)
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.frame_transform = frame_transform

        # --- Data Integrity Check (Optional but Recommended) ---
        # Check if all referenced files exist
        missing_files = []
        for idx, row in self.labels_df.iterrows():
            video_id = str(row['video_id']) # Ensure it's a string
            # Guess potential file extensions, adjust as needed
            potential_video_paths = [os.path.join(self.video_dir, f"{video_id}.mp4"),
                                     os.path.join(self.video_dir, f"{video_id}.avi"),
                                     os.path.join(self.video_dir, f"{video_id}.mov")]
            potential_audio_paths = [os.path.join(self.audio_dir, f"{video_id}.wav"),
                                     os.path.join(self.audio_dir, f"{video_id}.mp3")]

            if not any(os.path.exists(p) for p in potential_video_paths):
                 missing_files.append(f"Video for {video_id}")
            if not any(os.path.exists(p) for p in potential_audio_paths):
                 missing_files.append(f"Audio for {video_id}")

        if missing_files:
            print(f"Warning: Missing {len(missing_files)} files:")
            # for f in missing_files[:10]: print(f" - {f}") # Print first few
            # Consider filtering self.labels_df here or raising an error
            # self.labels_df = self.labels_df[~self.labels_df['video_id'].isin(get_ids_from_missing(missing_files))]


    def __len__(self):
        return len(self.labels_df)

    def _find_file(self, directory, file_id, extensions):
        """Helper to find file with different extensions."""
        for ext in extensions:
            path = os.path.join(directory, f"{file_id}{ext}")
            if os.path.exists(path):
                return path
        return None # Or raise error

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.labels_df.iloc[idx]
        video_id = str(row['video_id'])
        valence = row['valence']
        arousal = row['arousal']
        labels = torch.tensor([valence, arousal], dtype=torch.float32)

        # --- Find and Load Video Frames ---
        video_path = self._find_file(self.video_dir, video_id, ['.mp4', '.avi', '.mov'])
        if video_path is None:
             print(f"Error: Video file for ID {video_id} not found!")
             # Return dummy data or raise error - returning dummy for now
             dummy_frames = torch.zeros(3, hp.FRAME_SIZE[0], hp.FRAME_SIZE[1]) # C, H, W
             num_frames = 1
             # Ensure we also return dummy audio and valid labels structure
             dummy_waveform = torch.zeros(1, hp.TARGET_SR) # 1 channel, 1 second
             return {'video_frames': dummy_frames, 'num_frames': num_frames, 'audio_waveform': dummy_waveform, 'labels': labels, 'id': video_id}

        # Extract frames (returns list of NumPy arrays H,W,C BGR)
        # This can be slow - consider preprocessing offline if dataset is large
        frames_np = preprocessing.extract_frames(video_path, target_fps=hp.TARGET_FPS)

        if not frames_np: # If no frames extracted
             print(f"Warning: No frames extracted for video {video_id}")
             dummy_frames = torch.zeros(3, hp.FRAME_SIZE[0], hp.FRAME_SIZE[1])
             num_frames = 0
        else:
            # Apply transform to each frame (normalize, ToTensor)
            # Stack frames into a single tensor [NumFrames, Channels, Height, Width]
            frames_tensor_list = [self.frame_transform(frame) for frame in frames_np]
            if frames_tensor_list:
                 frames = torch.stack(frames_tensor_list)
                 num_frames = frames.shape[0]
            else: # Handle case where transforms might fail? Unlikely with ToTensor
                 dummy_frames = torch.zeros(3, hp.FRAME_SIZE[0], hp.FRAME_SIZE[1])
                 num_frames = 0

        # --- Find and Load Audio ---
        audio_path = self._find_file(self.audio_dir, video_id, ['.wav', '.mp3'])
        if audio_path is None:
             print(f"Error: Audio file for ID {video_id} not found!")
             dummy_waveform = torch.zeros(1, hp.TARGET_SR) # 1 channel, 1 second
        else:
             waveform, sr = preprocessing.load_and_resample_audio(audio_path, target_sr=hp.TARGET_SR)
             if waveform is None: # Handle loading error
                 print(f"Warning: Could not load audio for {video_id}")
                 waveform = torch.zeros(1, hp.TARGET_SR)

        # --- Return Sample ---
        # Note: Returning sequence of frames. Aggregation/pooling will happen later.
        # Or aggregate here if preferred (e.g., mean frame)
        # For now, return the sequence for flexibility
        sample = {
            'video_frames': frames if num_frames > 0 else dummy_frames, # [N_Frames, C, H, W] or [C,H,W]
            'num_frames': num_frames,
            'audio_waveform': waveform if waveform is not None else dummy_waveform, # [Channels, Time]
            'labels': labels, # [2]
            'id': video_id
        }

        return sample

# --- Custom Collate Function (if needed for padding/batching varying sequences) ---
# If you process sequences directly (not aggregating features in extractor/dataset)
# you might need a collate_fn for the DataLoader to handle varying numbers of frames
# or audio lengths per batch (e.g., padding).

# Example:
# def collate_emotion_data(batch):
#     # Custom logic to pad frames and waveforms to max length in batch
#     # ...
#     return processed_batch