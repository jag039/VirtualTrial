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
    
    loss_presence_topwear = criterion_presence(topwear_presence, labels['topwear_presence'])
    loss_presence_bottomwear = criterion_presence(bottomwear_presence, labels['bottomwear_presence'])
    loss_presence_both = criterion_presence(both_presence, labels['both_presence'])
    # SHOUOLD THIS LINE BE IN AN IF STATEMNT
    total_loss += loss_presence_topwear + loss_presence_bottomwear + loss_presence_both
    
    # 2. Category Loss (only compute if the clothing type is present)
    if labels['topwear_presence'].any():  # If topwear is present
        loss_category_topwear = criterion_category(topwear_category, labels['category_id'])
        total_loss += loss_category_topwear

    if labels['bottomwear_presence'].any():  # If bottomwear is present
        loss_category_bottomwear = criterion_category(bottomwear_category, labels['category_id'])
        total_loss += loss_category_bottomwear

    if labels['both_presence'].any():  # If both (dress) is present
        loss_category_both = criterion_category(both_category, labels['category_id'])
        total_loss += loss_category_both
    
    return total_loss