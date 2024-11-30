import torch

def infonce_loss(similarities, positive_indices, temperature=0.1):
    """
    Compute the InfoNCE loss using the log-sum-exp trick for numerical stability.

    Arguments:
    - similarities: A tensor of shape [batch_size, num_samples] containing similarity scores between samples.
    - positive_indices: A tensor of indices (shape [batch_size, 1]) that identifies the positive sample for each example in the batch.
    - temperature: A scalar to scale the logits.

    Returns:
    - loss: Computed InfoNCE loss.
    """
    # Scale the similarities by the temperature
    logits = similarities / temperature # [B, N]

    # sampled_cols = logits[:, positive_indices[:, 0]].T # [B, B]

    # all_logits = torch.cat([sampled_cols, logits], dim=1) # [B, B]

    # Find the max value in each row for numerical stability in log-sum-exp calculation
    max_logits = torch.max(logits, dim=1, keepdim=True)[0]

    # Subtract the max logit and exponentiate for stable softmax computation
    exp_logits = torch.exp(logits - max_logits)

    # Compute the sum of exp logits
    sum_exp_logits = torch.sum(exp_logits, dim=1, keepdim=True)

    # Log-sum-exp trick: log of sum of exps, adding back the max value subtracted earlier
    log_denominator = torch.log(sum_exp_logits) + max_logits

    # Gather the logits corresponding to positive pairs
    positive_logits = torch.gather(logits, 1, positive_indices)

    # Calculate log probabilities by subtracting log of denominator from positive logits
    log_prob = positive_logits - log_denominator

    # Compute mean of the negative log probabilities across the batch
    loss = -torch.mean(log_prob)

    return loss
