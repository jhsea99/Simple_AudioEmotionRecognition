# model/audio_feature_extractor.py
import torch
import torch.nn as nn
import torchaudio.transforms as T

import hyperparameters as hp

class AudioFeatureExtractor(nn.Module):
    """Extracts MFCC features from audio waveforms."""
    def __init__(self, target_sr=hp.TARGET_SR, n_mfcc=hp.N_MFCC,
                 win_length=hp.MFCC_WIN_LENGTH, hop_length=hp.MFCC_HOP_LENGTH,
                 aggregation=hp.FEATURE_AGGREGATION):
        super().__init__()
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self.aggregation = aggregation

        self.mfcc_transform = T.MFCC(
            sample_rate=target_sr,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': 400, # Common value
                'win_length': win_length,
                'hop_length': hop_length,
                'n_mels': 64, # Usually more mels than MFCCs
                'center': True,
            }
        )

    def forward(self, waveform):
        """
        Input: waveform (Tensor [Batch, Channels, Time] or [Channels, Time])
        Output: Aggregated MFCC features (Tensor [Batch, n_mfcc])
        """
        # Ensure waveform is on the correct device (although MFCC is often faster on CPU)
        # waveform = waveform.to(self.mfcc_transform.device) # Check if MFCC runs on GPU

        # Expects [Batch, Time] or [Time] for MFCC if mono
        if waveform.dim() == 3: # Batch, Channels, Time
            # Convert to mono if necessary, take first channel
            waveform = waveform[:, 0, :] # Take first channel -> [Batch, Time]
        elif waveform.dim() == 2: # Channels, Time
             waveform = waveform[0, :] # Take first channel -> [Time]
             waveform = waveform.unsqueeze(0) # Add batch dim -> [1, Time]
        # Add more checks if needed for shape [Time] input directly


        # Compute MFCCs -> Output shape: (Batch, n_mfcc, Time_frames)
        mfccs = self.mfcc_transform(waveform)

        # Aggregate over time dimension (dim=2)
        if self.aggregation == 'mean':
            features = torch.mean(mfccs, dim=2)
        elif self.aggregation == 'max':
            features, _ = torch.max(mfccs, dim=2)
        else: # Default to mean
             features = torch.mean(mfccs, dim=2)

        return features # Shape: [Batch, n_mfcc]

# Example usage (optional)
# if __name__ == '__main__':
#     extractor = AudioFeatureExtractor()
#     dummy_audio = torch.randn(4, 1, hp.TARGET_SR * 5) # Batch=4, 1 Channel, 5 seconds
#     features = extractor(dummy_audio)
#     print("Audio feature shape:", features.shape) # Should be [4, N_MFCC]