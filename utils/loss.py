import torch
import torch.nn as nn

criterion_category = nn.CrossEntropyLoss()  # Loss function for category classification
criterion_presence = nn.BCEWithLogitsLoss()  # Loss for binary presence classification

def compute_loss(preds, labels):
    """
    Compute the loss for category classification and binary presence detection.

    Args:
    - preds: Tuple of predictions 
        (presence_probs, topwear_category, bottomwear_category, both_category)
    - labels: Dictionary containing the ground truth labels:
        {
            'category_id': category ID (e.g., 1-50),
            'topwear_presence': binary label (0 or 1),
            'bottomwear_presence': binary label (0 or 1),
            'both_presence': binary label (0 or 1)
        }

    Returns:
    - total_loss: The combined loss (presence + category classification)
    """
    
    # Unpack predictions
    topwear_presence, bottomwear_presence, both_presence, topwear_category, bottomwear_category, both_category = preds
    total_loss = 0.0

    category_id = labels['category_id'].to(torch.long)  # Convert to long tensor for CrossEntropyLoss

    # Mask for valid samples
    topwear_mask = labels['topwear_presence'] > 0  # Get indices where topwear is present
    bottomwear_mask = labels['bottomwear_presence'] > 0  # Get indices where bottomwear is present
    both_mask = labels['both_presence'] > 0  # Get indices where both is present

    # Compute category loss only for relevant samples
    if topwear_mask.any():
        loss_category_topwear = criterion_category(topwear_category[topwear_mask], category_id[topwear_mask])
        loss_presence_topwear = criterion_presence(topwear_presence[topwear_mask], labels['topwear_presence'][topwear_mask])
        total_loss += loss_category_topwear + loss_presence_topwear

    if bottomwear_mask.any():
        loss_category_bottomwear = criterion_category(bottomwear_category[bottomwear_mask], category_id[bottomwear_mask])
        loss_presence_bottomwear = criterion_presence(bottomwear_presence[bottomwear_mask], labels['bottomwear_presence'][bottomwear_mask])
        total_loss += loss_category_bottomwear + loss_presence_bottomwear

    if both_mask.any():
        loss_category_both = criterion_category(both_category[both_mask], category_id[both_mask])
        loss_presence_both = criterion_presence(both_presence[both_mask], labels['both_presence'][both_mask])
        total_loss += loss_category_both + loss_presence_both
    
    return total_loss