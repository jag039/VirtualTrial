import torch
import torch.nn as nn

def compute_loss(preds, targets):
    """
    Computes the cross-entropy loss for garment classification.

    Parameters:
    - preds (torch.Tensor): The output logits from the model with shape (batch_size, num_classes).
    - targets (dict): A dictionary containing the key 'category_id', a tensor of shape (batch_size,)
                        representing the ground-truth class indices.

    Returns:
    - loss (torch.Tensor): The computed cross-entropy loss.
    """
    # Extract ground-truth labels
    category_id = targets['category_id']
    
    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Compute and return the loss
    loss = loss_fn(preds, category_id)
    return loss