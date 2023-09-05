import torch

def calculate_ppr(A, alpha=0.5, topk=None, tol=1e-6, device='cuda'):
    A = A.float().to(device)
    alpha = torch.tensor(alpha, dtype=torch.float, device=device)
    I = torch.eye(A.size(0), device=device)
    
    # Create a normalized adjacency matrix
    A=A.double()
    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.rsqrt(deg + 1e-10)
    A_hat = A * deg_inv_sqrt.unsqueeze(1) * deg_inv_sqrt.unsqueeze(0)
    A_hat = A_hat + I  # add self-loops

    # Initial guess for P
    P = torch.full(A_hat.shape, 1/A.size(0), device=device).double()
    diff = float('inf')
    while diff > tol:
        new_P = (1 - alpha) * torch.mm(A_hat, P) + alpha * I
        diff = torch.abs(P - new_P).sum()
        P = new_P

    if topk is not None:
        P_topk_val, P_topk_idx = torch.topk(P, topk, dim=1)
        P.fill_(0)
        P.scatter_(1, P_topk_idx, P_topk_val)
        
    return P
