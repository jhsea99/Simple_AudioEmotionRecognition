# emotion_predictor.py

import torch
import torch.nn as nn
from hyperparameters import FEATURE_VECTOR_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT

class EmotionPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(EmotionPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2) # Output: Valence and Arousal (2 values)
        )

    def forward(self, x):
        return self.mlp(x)

# --- Optional: MLP with Transformer ---
class EmotionPredictorWithTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_head, dropout, transformer_dim_feedforward):
        super(EmotionPredictorWithTransformer, self).__init__()
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_head,
                                                                    dim_feedforward=transformer_dim_feedforward,
                                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)

        self.linear_out = nn.Linear(hidden_size, 2) # Output: Valence and Arousal

    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.dropout(x)

        # The input to the Transformer needs to have a sequence dimension.
        # We can add a dummy sequence length of 1.
        x = x.unsqueeze(1) # Shape: (batch_size, 1, hidden_size)
        transformer_output = self.transformer_encoder(x) # Shape: (batch_size, 1, hidden_size)
        transformer_output = transformer_output.squeeze(1) # Shape: (batch_size, hidden_size)

        output = self.linear_out(transformer_output)
        return output

if __name__ == '__main__':
    # Example usage for simple MLP
    if FEATURE_VECTOR_SIZE is None:
        print("Warning: FEATURE_VECTOR_SIZE is not defined in hyperparameters. Please define it based on your ResNet and MFCC output.")
    else:
        dummy_input = torch.randn(2, FEATURE_VECTOR_SIZE)
        mlp_model = EmotionPredictor(FEATURE_VECTOR_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
        mlp_output = mlp_model(dummy_input)
        print("MLP output shape:", mlp_output.shape)

        # Example usage for MLP with Transformer
        transformer_model = EmotionPredictorWithTransformer(FEATURE_VECTOR_SIZE, HIDDEN_SIZE, NUM_LAYERS, 4, DROPOUT, 512)
        transformer_output = transformer_model(dummy_input)
        print("Transformer output shape:", transformer_output.shape)