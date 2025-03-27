# dataset.py

import os
import torch
import torchaudio
from torch.utils.data import Dataset
from PIL import Image # For loading images
import pandas as pd # If your annotations are in a CSV file
from hyperparameters import SAMPLE_RATE, ALIGNED_FACE_SIZE

class FaceAudioEmotionDataset(Dataset):
    def __init__(self, audio_dir, face_image_dir, annotation_file):
        """
        Args:
            audio_dir (string): Directory containing audio files.
            face_image_dir (string): Directory containing aligned face image files (or subdirectories).
            annotation_file (string): Path to the annotation file (e.g., CSV).
        """
        self.audio_dir = audio_dir
        self.face_image_dir = face_image_dir
        self.annotations = self._load_annotations(annotation_file)
        self.audio_files = list(self.annotations.keys()) # Assuming the keys in annotations correspond to audio file names (without extension)

    def _load_annotations(self, annotation_file):
        """
        Loads the annotations (Valence, Arousal) from the annotation file.
        This needs to be adapted based on your annotation file format.

        Example for a CSV file where the first column is the 'audio_id'
        and the columns 'valence' and 'arousal' exist:
        """
        annotations = {}
        try:
            df = pd.read_csv(annotation_file)
            for index, row in df.iterrows():
                audio_id = row['audio_id'] # Adjust the column name as needed
                valence = row['valence']   # Adjust the column name as needed
                arousal = row['arousal']   # Adjust the column name as needed
                annotations[audio_id] = {'valence': valence, 'arousal': arousal}
        except FileNotFoundError:
            print(f"Error: Annotation file not found at {annotation_file}")
        return annotations

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_id = self.audio_files[idx]

        # Load audio
        audio_path = os.path.join(self.audio_dir, f"{audio_id}.wav") # Assuming .wav format, adjust if needed
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
                waveform = resampler(waveform)
            # Ensure mono audio
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        except FileNotFoundError:
            print(f"Error: Audio file not found at {audio_path}")
            return None # Or handle the error as appropriate

        # Load face image
        face_image_path = os.path.join(self.face_image_dir, f"{audio_id}.jpg") # Assuming .jpg format and same naming, adjust if needed
        try:
            face_image = Image.open(face_image_path).convert('RGB')
            # Resize the image if necessary
            if face_image.size != ALIGNED_FACE_SIZE:
                face_image = face_image.resize(ALIGNED_FACE_SIZE)
            face_image = torch.tensor(np.array(face_image)).float().permute(2, 0, 1) / 255.0 # Convert to tensor and normalize
        except FileNotFoundError:
            print(f"Error: Face image not found at {face_image_path}")
            return None # Or handle the error as appropriate

        # Get annotations
        target = torch.tensor([self.annotations[audio_id]['valence'], self.annotations[audio_id]['arousal']]).float()

        return waveform, face_image, target

if __name__ == '__main__':
    # Example usage (replace with your actual paths)
    AUDIO_DIR = '/path/to/your/audio/data'
    FACE_IMAGE_DIR = '/path/to/your/aligned_face_images'
    ANNOTATION_FILE = '/path/to/your/annotations.csv'

    # Create dummy files and annotations for testing
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(FACE_IMAGE_DIR, exist_ok=True)
    with open(ANNOTATION_FILE, 'w') as f:
        f.write("audio_id,valence,arousal\n")
        f.write("audio_1,0.5,0.6\n")
        f.write("audio_2,-0.2,0.8\n")

    # Create dummy audio files
    dummy_audio_1 = torch.randn(1, SAMPLE_RATE * 2) # 2 seconds of mono audio
    torchaudio.save(os.path.join(AUDIO_DIR, 'audio_1.wav'), dummy_audio_1, SAMPLE_RATE)
    dummy_audio_2 = torch.randn(1, SAMPLE_RATE * 3) # 3 seconds of mono audio
    torchaudio.save(os.path.join(AUDIO_DIR, 'audio_2.wav'), dummy_audio_2, SAMPLE_RATE)

    # Create dummy image files
    dummy_image_1 = Image.new('RGB', (128, 128), color = 'red')
    dummy_image_1_resized = dummy_image_1.resize(ALIGNED_FACE_SIZE)
    dummy_image_1_resized.save(os.path.join(FACE_IMAGE_DIR, 'audio_1.jpg'))
    dummy_image_2 = Image.new('RGB', (64, 64), color = 'blue')
    dummy_image_2_resized = dummy_image_2.resize(ALIGNED_FACE_SIZE)
    dummy_image_2_resized.save(os.path.join(FACE_IMAGE_DIR, 'audio_2.jpg'))

    dataset = FaceAudioEmotionDataset(AUDIO_DIR, FACE_IMAGE_DIR, ANNOTATION_FILE)
    print(f"Number of samples in the dataset: {len(dataset)}")

    # Get an example item
    if len(dataset) > 0:
        waveform, face_image, target = dataset[0]
        print("Shape of audio waveform:", waveform.shape)
        print("Shape of face image tensor:", face_image.shape)
        print("Target Valence and Arousal:", target)