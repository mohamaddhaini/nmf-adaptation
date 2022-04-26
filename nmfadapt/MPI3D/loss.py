import numpy as np 
import torch
from sklearn.decomposition._nmf import _initialize_nmf
import math
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh


def sparse(Y,M,mu,lamda,iterations):
    with torch.no_grad():
      rank=M.shape[1]
      features=Y.shape[1]
      loss=[]
      # A=(torch.zeros(rank,features)).cuda()
      A=(torch.rand(rank,features)).cuda()
    for i in range(0,iterations):
        A = A*torch.matmul(M.T,Y)/(torch.matmul(M.T,M.mm(A))+lamda*torch.ones(rank,features).cuda()+mu*A)
        A[A < 1e-5] = 0
        # A = A/torch.norm(A,p=2)
        res=Y-torch.matmul(M,A)
        loss.append(torch.norm(res,p=2).detach().cpu())
    return res,M,A,np.array(loss)

def nmf_mu(Y,rank,mu,lamda,iterations,M1=None):
    with torch.no_grad():
      features=Y.shape[1]
      batch=Y.shape[0]
      if M1 is None:
        M1, A1 = _initialize_nmf(Y.data.cpu().numpy(), rank, init='random')
        M=torch.from_numpy(M1).cuda()
        A=torch.from_numpy(A1).cuda()
      else:
        _, A1 = _initialize_nmf(Y.data.cpu().numpy(), rank, init='random')
        M=torch.nn.Parameter(M1).cuda()
        A=torch.from_numpy(A1).cuda()
      loss=[]
      # while torch.norm(Y-M.mm(A),p='fro')>2.9:
    for i in range(0,iterations):
        M = M* ((Y.mm(A.T))/(torch.matmul(M,A.mm(A.T))+mu*M))
        M[M < 1e-16] = 1e-16
        A = A*torch.matmul(M.T,Y)/(torch.matmul(M.T,M.mm(A))+lamda*torch.ones(rank,features).cuda())
        res=Y-torch.matmul(M,A)
        loss.append(torch.norm(res,p=2).detach().cpu())
    return np.array(loss),res,M,A,M.data


def _lipschitz_constant(W):
    #L = torch.linalg.norm(W, ord=2) ** 2
    WtW = torch.matmul(W.t(), W)
    #L = torch.linalg.eigvalsh(WtW)[-1]
    L = eigsh(WtW.detach().cpu().numpy(), k=1, which='LM',
              return_eigenvectors=False).item()
    return L

def ridge(b, A, alpha=1e-4):
    # right-hand side
    rhs = torch.matmul(A.T, b)
    # regularized gram matrix
    M = torch.matmul(A.T, A)
    M.diagonal().add_(alpha)
    # solve
    L, info = torch.linalg.cholesky_ex(M)
    if info != 0:
        raise RuntimeError("The Gram matrix is not positive definite. "
                           "Try increasing 'alpha'.")
    x = torch.cholesky_solve(rhs, L)
    return x

def ista(x, weight, alpha=1.0, fast=True, lr='auto', maxiter=100,
         tol=1e-5, verbose=False):
    
    n_samples = x.size(0)
    n_components = weight.size(1)
    z0 = ridge(x.T, weight, alpha=alpha).T
    # z0 = x.new_zeros(n_samples, n_components)
    if lr == 'auto':
        # set lr based on the maximum eigenvalue of W^T @ W; i.e. the
        # Lipschitz constant of \grad f(z), where f(z) = ||Wz - x||^2
        L = _lipschitz_constant(weight)
        lr = 1 / L
    tol = z0.numel() * tol

    def loss_fn(z_k):
        resid = torch.matmul(z_k, weight.T) - x
        loss = 0.5 * resid.pow(2).sum() + alpha * z_k.abs().sum()
        return loss / x.size(0)

    def rss_grad(z_k):
        resid = torch.matmul(z_k, weight.T) - x
        return torch.matmul(resid, weight)
    
    # optimize
    z = z0
    if fast:
        y, t = z0, 1
    loss=[]
    for _ in range(maxiter):
        if verbose:
            print('loss: %0.4f' % loss_fn(z))

        # ista update
        z_prev = y if fast else z
        
        z_next = F.softshrink(z_prev - lr * rss_grad(z_prev), alpha * lr)

        # check convergence
        if (z - z_next).abs().sum() <= tol:
            z = z_next
            break

        # update variables
        if fast:
            t_next = (1 + math.sqrt(1 + 4 * t**2)) / 2
            y = z_next + ((t-1)/t_next) * (z_next - z)
            t = t_next
        z = z_next
        loss.append(loss_fn(z).clone().detach().cpu().numpy())
    return z,loss

def dictionary_learning(max_iter,R,rank,lambda_sp,lambda_reg,lamda = 0.000001):
    with torch.no_grad():
        # u, s, v = torch.svd(R)
        # M=u[:,:rank].float()
        # A ,_= ista(R.T,M,alpha=lambda_sp,maxiter=10)
        M=torch.randn(R.shape[0],rank).cuda()
        M = M/torch.norm(M,dim=0,p=2)
        A=torch.randn(R.shape[1],rank).cuda()
        loss=[]
        v = torch.zeros(R.shape[0],rank).cuda()
    for i in range(max_iter):
        error = R - torch.matmul(M, A.T)
        #Batch Update
        v= lambda_reg*v - lamda * (torch.matmul(error, A))
        M=M-v
        M = M/torch.norm(M,dim=0,p=2)
        A,_ = ista(R.T,M,alpha=lambda_sp,maxiter=1)
        loss.append(torch.linalg.norm(error).cpu().detach().numpy())
    error = R - torch.matmul(M, A.T)
    return M,A,error,np.array(loss)
