import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import importlib
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from sklearn.decomposition._nmf import _initialize_nmf
os.chdir(r'path_to_project_folder')
import model
import transform as tran
import argparse
import time
torch.set_num_threads(1)
from read_data import ImageList_r as ImageList
import matplotlib.pyplot as plt 
import random
#%%
os.chdir(r'path_to_data')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(0)
np.random.seed(0)
def set_seed():
    torch.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(3)
    random.seed(3)
    os.environ['PYTHONHASHSEED'] = str(3)

set_seed()

torch.set_default_tensor_type(torch.FloatTensor)
use_gpu = True
if use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data_transforms = {
    'train': tran.rr_train(resize_size=224),
    'val': tran.rr_train(resize_size=224),
    'test': tran.rr_eval(resize_size=224),
}
# set dataset
batch_size = {"train": 36, "val": 36, "test": 4}

c="color.txt"
n="noisy.txt"
s="scream.txt"

c_t="color_test.txt"
n_t="noisy_test.txt"
s_t="scream_test.txt"


dsets = {"train": ImageList(open(c).readlines(), transform=data_transforms["train"]),
         "val": ImageList(open(n).readlines(),transform=data_transforms["val"]),
         "test": ImageList(open(n_t).readlines(),transform=data_transforms["test"])}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],
                                               shuffle=True, num_workers=0)
                for x in ['train', 'val']}
dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=batch_size["test"],
                                                   shuffle=False, num_workers=0)

dset_sizes = {x: len(dsets[x]) for x in ['train', 'val','test']}
# device = torch.device('cuda')
#%%
def set_seed():
    torch.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(3)
    random.seed(3)
    os.environ['PYTHONHASHSEED'] = str(3)

set_seed()



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

def match_nmf(Feature_s, Feature_t):
    _,res1,b_s,A_s,R_s=nmf_mu(Feature_s.T,18,0,0,100)
    loss,res2,b_t,A_t,R_t=nmf_mu(Feature_t.T,18,0,0,100)
    res_s,_,aa,_=sparse(b_s,b_t,0,0,100)
    res_t,_,_,loss=sparse(b_t,b_s,0,0,100)
    return torch.norm(res_s,p='fro')+torch.norm(res_t,p='fro')

def Regression_test(loader, model):
    MSE = [0, 0, 0, 0]
    MAE = [0, 0, 0, 0]
    number = 0
    with torch.no_grad():
        for (imgs, labels) in loader['test']:
            imgs = imgs.to(device)
            labels_source = labels.to(device)
            labels1 = labels_source[:, 0]
            labels3 = labels_source[:, 2]
            labels4 = labels_source[:, 3]
            labels1 = labels1.unsqueeze(1)
            labels3 = labels3.unsqueeze(1)
            labels4 = labels4.unsqueeze(1)
            labels_source = torch.cat((labels1, labels3, labels4), dim=1)
            labels = labels_source.float()
            pred = model(imgs)
            MSE[0] += torch.nn.MSELoss(reduction='sum')(pred[:, 0], labels[:, 0])
            MAE[0] += torch.nn.L1Loss(reduction='sum')(pred[:, 0], labels[:, 0])
            MSE[1] += torch.nn.MSELoss(reduction='sum')(pred[:, 1], labels[:, 1])
            MAE[1] += torch.nn.L1Loss(reduction='sum')(pred[:, 1], labels[:, 1])
            MSE[2] += torch.nn.MSELoss(reduction='sum')(pred[:, 2], labels[:, 2])
            MAE[2] += torch.nn.L1Loss(reduction='sum')(pred[:, 2], labels[:, 2])
            MSE[3] += torch.nn.MSELoss(reduction='sum')(pred, labels)
            MAE[3] += torch.nn.L1Loss(reduction='sum')(pred, labels)
            number += imgs.size(0)
    for j in range(4):
        MSE[j] = MSE[j] / number
        MAE[j] = MAE[j] / number
    print("\tMSE : {0},{1},{2}\n".format(MSE[0],MSE[1],MSE[2]))
    print("\tMAE : {0},{1},{2}\n".format(MAE[0], MAE[1], MAE[2]))
    print("\tMSEall : {0}\n".format(MSE[3]))
    print("\tMAEall : {0}\n".format(MAE[3]))
    return MAE[3]

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer

class Model_Regression(nn.Module):
    def __init__(self):
        super(Model_Regression,self).__init__()
        self.model_fc = model.Resnet18Fc()
        self.classifier_layer = nn.Linear(512, 3)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        self.classifier_layer = nn.Sequential(self.classifier_layer,  nn.Sigmoid())
        self.predict_layer = nn.Sequential(self.model_fc,self.classifier_layer)
    def forward(self,x):
        feature = self.model_fc(x)
        outC= self.classifier_layer(feature)
        return(outC,feature)

test_graph_nmf=[]
list1=[]
list2=[]
path=r'path_to_save_results'

set_seed()
name='nmf'
loss_graph=[]
A1=torch.nn.Parameter(torch.rand(36,36))
A2=torch.nn.Parameter(torch.rand(36,36))
Model_R = Model_Regression()
Model_R = Model_R.to(device)
Model_R.train(True)
criterion = {"regressor": nn.MSELoss()}
optimizer_dict = [{"params": filter(lambda p: p.requires_grad, Model_R.model_fc.parameters()), "lr": 0.1},
                  {"params": filter(lambda p: p.requires_grad, Model_R.classifier_layer.parameters()), "lr": 1},{'params':A1,'lr':1},{'params':A2,'lr':1}]
optimizer = optim.SGD(optimizer_dict,lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
train_cross_loss = train_nmf_loss = train_total_loss = 0.0
len_source = len(dset_loaders["train"]) - 1
print(len_source)
len_target = len(dset_loaders["val"]) - 1
param_lr = []
iter_source = iter(dset_loaders["train"])
iter_target = iter(dset_loaders["val"])
for param_group in optimizer.param_groups:
    param_lr.append(param_group["lr"])
test_interval = 500
print_interval = 100
num_iter = 3*len_source
start=time.time()
test_init=np.inf
beta=1
rank=128
for iter_num in range(1, num_iter + 1):
    Model_R.train(True)
    optimizer = inv_lr_scheduler(param_lr, optimizer , iter_num, gamma=0.0001, power=0.75,init_lr=0.1,weight_decay=0.0005)
    optimizer.zero_grad()
    if iter_num % len_source == 0:
        iter_source = iter(dset_loaders["train"])
    if iter_num % len_target == 0:
        iter_target = iter(dset_loaders["val"])
    data_source = iter_source.next()
    data_target = iter_target.next()
    inputs_source, labels_source = data_source
    labels1 = labels_source[:, 0]
    labels3 = labels_source[:, 2]
    labels4 = labels_source[:, 3]
    labels1 = labels1.unsqueeze(1)
    labels3 = labels3.unsqueeze(1)
    labels4 = labels4.unsqueeze(1)
    labels_source = torch.cat((labels1,labels3,labels4),dim=1)
    labels_source = labels_source.float()
    inputs_target, labels_target = data_target
    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    inputs = inputs.to(device)
    labels = labels_source.to(device)
    inputs_s = inputs.narrow(0, 0, batch_size["train"])
    inputs_t = inputs.narrow(0, batch_size["train"], batch_size["train"])
    outC_s, feature_s = Model_R(inputs_s)
    outC_t, feature_t = Model_R(inputs_t)
    classifier_loss = criterion["regressor"](outC_s, labels)
    nmf_loss= match_nmf(feature_s,feature_t)
    total_loss = classifier_loss + 0.001*nmf_loss
    total_loss.backward()
    optimizer.step()
    train_cross_loss += classifier_loss.item()
    train_nmf_loss += nmf_loss.item()
    train_total_loss += total_loss.item()
    list2.append(time.time()-start)
    if (iter_num % print_interval) == 0:
        print("Iter {:05d}, Average Cross Entropy Loss: {:.4f}; Average NMF Loss: {:.4f};  Average Training Loss: {:.4f};LR:{:.6f}".format(
            iter_num, train_cross_loss / float(test_interval), train_nmf_loss / float(test_interval),
            train_total_loss / float(test_interval),optimizer.param_groups[0]['lr']))
        loss_graph.append(train_nmf_loss / float(test_interval))
        train_cross_loss = train_nmf_loss = train_total_loss  = 0.0
    if (iter_num % test_interval) == 0:
        Model_R.eval()
        test_loss=Regression_test(dset_loaders, Model_R.predict_layer)
        if test_loss<test_init:
            test_init=test_loss
            print('Saving')
            torch.save(Model_R.state_dict(), os.path.join(path,name))
