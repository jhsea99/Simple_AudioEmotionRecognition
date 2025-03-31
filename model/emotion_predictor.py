# model/emotion_predictor.py
import torch
import torch.nn as nn

import hyperparameters as hp

class EmotionPredictorMLP(nn.Module):
    """MLP to predict Valence and Arousal from concatenated features."""
    def __init__(self, input_dim=hp.FACE_FEATURE_DIM + hp.AUDIO_FEATURE_DIM,
                 hidden_layers=hp.MLP_HIDDEN_LAYERS, output_dim=hp.OUTPUT_DIM):
        super().__init__()
        self.input_dim = input_dim
        layers = []
        current_dim = input_dim

        # Input layer -> First hidden layer
        layers.append(nn.Linear(current_dim, hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3)) # Add dropout for regularization
        current_dim = hidden_layers[0]

        # Additional hidden layers
        for hidden_dim in hidden_layers[1:]:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        # Output activation: Linear for regression. Tanh *can* be used if labels are scaled to [-1, 1]
        # layers.append(nn.Tanh()) # Optional: if your V/A labels are in [-1, 1] range

        self.network = nn.Sequential(*layers)

    def forward(self, face_features, audio_features, use_face=True):
        """
        Forward pass. Handles training (both features) and testing (audio only).

        Args:
            face_features (Tensor): Features from face extractor [Batch, FACE_FEATURE_DIM]
                                   Can be None or zeros during audio-only inference.
            audio_features (Tensor): Features from audio extractor [Batch, AUDIO_FEATURE_DIM]
            use_face (bool): Flag indicating if face features should be used.

        Returns:
            Tensor: Predictions for Valence and Arousal [Batch, OUTPUT_DIM]
        """
        if use_face:
            # Training or validation with both modalities
            if face_features is None:
                 raise ValueError("Face features cannot be None when use_face is True")
            # Ensure face features are correctly shaped (e.g., aggregated if needed)
            # Assuming face_features is already [Batch, FACE_FEATURE_DIM]
            combined_features = torch.cat((face_features, audio_features), dim=1)
        else:
            # Testing/Inference with audio only
            # Create dummy zero tensor for face features
            batch_size = audio_features.shape[0]
            # Ensure dummy features are on the same device as audio features
            dummy_face_features = torch.zeros(batch_size, hp.FACE_FEATURE_DIM,
                                              device=audio_features.device,
                                              dtype=audio_features.dtype)
            combined_features = torch.cat((dummy_face_features, audio_features), dim=1)

        # Check if combined feature dimension matches expected input dimension
        if combined_features.shape[1] != self.input_dim:
             raise ValueError(f"Combined feature dimension ({combined_features.shape[1]}) "
                              f"does not match MLP input dimension ({self.input_dim})")

        predictions = self.network(combined_features)
        return predictions