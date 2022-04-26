import numpy as np 
import torch
from sklearn.decomposition._nmf import _initialize_nmf
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
