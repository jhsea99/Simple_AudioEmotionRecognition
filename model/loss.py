import torch.nn as nn
from hyperparameters import LOSS_TYPE

if LOSS_TYPE == 'MSELoss':
    criterion = nn.MSELoss()
elif LOSS_TYPE == 'L1Loss':
    criterion = nn.L1Loss()
else:
    raise ValueError(f"Unsupported loss type: {LOSS_TYPE}. Choose from ['MSELoss', 'L1Loss']")