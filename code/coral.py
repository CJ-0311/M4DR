import torch
def CORAL(source, target):
    """
    Computes the CORAL loss between source and target features.
    
    Args:
        source (torch.Tensor): Source domain features, shape [batch_size, feature_dim].
        target (torch.Tensor): Target domain features, shape [batch_size, feature_dim].
        
    Returns:
        torch.Tensor: The CORAL loss.
    """
    d = source.size(1)  # feature dimension
    ns, nt = source.size(0), target.size(0)

    # Source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm) / (ns - 1)

    # Target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt) / (nt - 1)

    # Frobenius norm
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4 * d * d)
    
    return loss