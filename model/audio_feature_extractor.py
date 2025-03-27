# audio_feature_extractor.py

import torch
import torchaudio
from hyperparameters import SAMPLE_RATE, N_MFCC, MFCC_WINDOW_SIZE, MFCC_HOP_LENGTH

class AudioFeatureExtractor:
    def __init__(self):
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            melkwargs={
                'n_fft': int(SAMPLE_RATE * MFCC_WINDOW_SIZE),
                'hop_length': int(SAMPLE_RATE * MFCC_HOP_LENGTH),
                'n_mels': 23,  # Recommended value
            }
        )

    def extract_features(self, audio_waveform):
        """
        Extracts MFCC features from an audio waveform.

        Args:
            audio_waveform (torch.Tensor): The audio waveform as a PyTorch tensor
                                          (shape: (num_channels, num_samples)).
                                          For mono audio, num_channels is 1.

        Returns:
            torch.Tensor: The MFCC features (shape: (batch_size, n_mfcc, time_frames)).
                          We will typically take the mean or flatten these features.
        """
        # Ensure the waveform is mono (if it's stereo, you might want to average channels)
        if audio_waveform.shape[0] > 1:
            audio_waveform = torch.mean(audio_waveform, dim=0, keepdim=True)

        mfcc_features = self.mfcc_transform(audio_waveform)
        return mfcc_features

if __name__ == '__main__':
    # Example usage
    import numpy as np
    dummy_audio = torch.randn(1, SAMPLE_RATE * 3) # 3 seconds of mono audio at the defined sample rate
    extractor = AudioFeatureExtractor()
    mfcc_output = extractor.extract_features(dummy_audio)
    print("MFCC output shape:", mfcc_output.shape) # Expected shape: (1, N_MFCC, time_frames)

    # You might want to process this output further, e.g., take the mean over time
    mean_mfcc = torch.mean(mfcc_output, dim=2)
    print("Mean MFCC shape:", mean_mfcc.shape) # Expected shape: (1, N_MFCC)