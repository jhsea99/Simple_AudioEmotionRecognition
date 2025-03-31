# model/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Concordance Correlation Coefficient (CCC) ---
# Often preferred for Valence-Arousal prediction
def concordance_correlation_coefficient(y_true, y_pred, reduction='mean'):
    """
    Computes the Concordance Correlation Coefficient (CCC).
    Assumes y_true and y_pred are Tensors of shape (N,).
    Based on Lin's CCC.
    """
    if y_true.dim() > 1 and y_true.shape[1] > 1: # Handle multi-output case column-wise
        return torch.stack([
            concordance_correlation_coefficient(y_true[:, i], y_pred[:, i], reduction)
            for i in range(y_true.shape[1])
        ]).mean() # Average CCC across outputs (V and A)


    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    var_true = torch.var(y_true, unbiased=False) # Population variance
    var_pred = torch.var(y_pred, unbiased=False) # Population variance
    std_true = torch.sqrt(var_true)
    std_pred = torch.sqrt(var_pred)
    # Covariance calculation
    covar = torch.mean((y_true - mean_true) * (y_pred - mean_pred))

    # CCC formula
    numerator = 2 * covar
    denominator = var_true + var_pred + (mean_true - mean_pred)**2

    # Handle potential division by zero
    # If denominator is close to zero, it means variance is zero or means are equal.
    # If variance is zero, correlation is undefined (or 1 if means match exactly).
    # We can return 0 or 1 based on the context, or handle eps. Let's return 0 for safety.
    eps = 1e-8
    ccc = torch.where(denominator > eps, numerator / denominator, torch.tensor(0.0, device=y_true.device))

    # CCC loss is often defined as 1 - CCC
    # loss = 1.0 - ccc

    # If reduction is 'none', return CCC per element (not typical use here)
    # if reduction == 'none':
    #     return ccc # Or loss = 1.0 - ccc
    # elif reduction == 'mean':
    #     return ccc.mean() # Or loss = 1.0 - ccc.mean()
    # elif reduction == 'sum':
    #      return ccc.sum() # Or loss = 1.0 - ccc.sum()

    # Return the CCC value itself, the loss function using it will do 1 - CCC if needed
    return ccc

class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """Calculates 1 - CCC for loss."""
        # y_pred and y_true shape: [Batch, OutputDim] (e.g., [16, 2])
        # Calculate CCC for each dimension (Valence, Arousal) separately
        ccc_v = concordance_correlation_coefficient(y_true[:, 0], y_pred[:, 0])
        ccc_a = concordance_correlation_coefficient(y_true[:, 1], y_pred[:, 1])
        # Average CCC loss (common practice)
        ccc_loss = 1.0 - (ccc_v + cca_a) / 2.0
        return ccc_loss

# --- Standard Loss Functions ---
def get_loss_function(loss_name=hp.LOSS_FUNCTION):
    """Returns the selected loss function."""
    if loss_name == 'MSE':
        return nn.MSELoss()
    elif loss_name == 'MAE':
        return nn.L1Loss() # MAE is L1Loss
    elif loss_name == 'CCC':
         print("Using CCC Loss (1 - mean(CCC_V, CCC_A))")
         # return CCCLoss() # Use this if you implement the CCCLoss class above
         # Simpler: return a lambda that calculates it directly
         return lambda pred, target: 1.0 - concordance_correlation_coefficient(target, pred)
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

# --- Evaluation Metrics ---
# Can reuse CCC function for evaluation
def calculate_metrics(y_pred, y_true, metrics_list=hp.EVALUATION_METRICS):
    """Calculates specified evaluation metrics."""
    results = {}
    # Ensure inputs are detached and on CPU for sklearn/numpy if needed
    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()

    # Ensure inputs are Tensors for torch metrics
    y_pred_t = y_pred.detach()
    y_true_t = y_true.detach()


    for metric in metrics_list:
        if metric == 'MSE':
            results['MSE_V'] = F.mse_loss(y_pred_t[:, 0], y_true_t[:, 0]).item()
            results['MSE_A'] = F.mse_loss(y_pred_t[:, 1], y_true_t[:, 1]).item()
            results['MSE_AVG'] = (results['MSE_V'] + results['MSE_A']) / 2.0
        elif metric == 'MAE':
            results['MAE_V'] = F.l1_loss(y_pred_t[:, 0], y_true_t[:, 0]).item()
            results['MAE_A'] = F.l1_loss(y_pred_t[:, 1], y_true_t[:, 1]).item()
            results['MAE_AVG'] = (results['MAE_V'] + results['MAE_A']) / 2.0
        elif metric == 'CCC':
             # Use the CCC function directly (requires tensors)
             ccc_v = concordance_correlation_coefficient(y_true_t[:, 0], y_pred_t[:, 0]).item()
             ccc_a = concordance_correlation_coefficient(y_true_t[:, 1], y_pred_t[:, 1]).item()
             results['CCC_V'] = ccc_v
             results['CCC_A'] = ccc_a
             results['CCC_AVG'] = (ccc_v + ccc_a) / 2.0 # Average CCC is a common single metric

    return results