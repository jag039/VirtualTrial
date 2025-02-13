import torch

def accuracy(preds, targets):
    """
    Computes the number of correct predictions in a batch.

    Args:
        preds (torch.Tensor): The model output logits of shape (batch_size, num_classes).
        targets (torch.Tensor): The ground-truth class indices of shape (batch_size,).

    Returns:
        int: The number of correct predictions.
    """
    predicted_labels = torch.argmax(preds, dim=1)
    correct = (predicted_labels == targets).sum().item()
    return correct