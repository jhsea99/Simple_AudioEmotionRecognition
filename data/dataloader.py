from torch.utils.data import DataLoader
from dataset import FaceAudioEmotionDataset
from hyperparameters import AUDIO_PATH, FACE_IMAGE_DIR, ANNOTATION_PATH, BATCH_SIZE

# Instantiate the dataset
dataset = FaceAudioEmotionDataset(AUDIO_PATH, FACE_IMAGE_DIR, ANNOTATION_PATH)

# Create the data loader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) # Adjust num_workers as needed

# Example of iterating through the data loader
if __name__ == '__main__':
    for batch in dataloader:
        audio_waveforms, face_images, targets = batch
        print("Audio batch shape:", audio_waveforms.shape)
        print("Face image batch shape:", face_images.shape)
        print("Target batch shape:", targets.shape)
        break # Just print the first batch for example