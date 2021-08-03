import torch

def nll(self, output, target):
    """Computes the negative log-likelihood for classification problem
        (cross-entropy).
    Args:
        output: float torch tensor, shape [batch_size, num_classes],
            The predictions of probabilites for each classes
        target: int torch tensor, shape [batch_size, 1], the corresponding
            labels.
    Returns:
        torch tensor: the resulting negative log-likelihood.
    """
    with torch.no_grad():
        result = -torch.log(output)[range(target.shape[0]), target].mean()
    return result

def accuracy(output, target):
    """Computes the accuracy of predictions.
    Args:
        output: float torch tensor, shape [batch_size, num_classes],
            The predictions of probabilites for each classes
        target: int torch tensor, shape [batch_size, 1], the corresponding
            labels.
    Returns:
        torch tensor: the resulting accuracy.
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)