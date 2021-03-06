import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import importlib
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import model
import loss as ls
import transform as tran
import argparse
import time
torch.set_num_threads(1)
import matplotlib.pyplot as plt 
import random
from read_data import ImageList_r as ImageList
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='PyTorch DAregre experiment')
parser.add_argument('--src', type=str, default='rc', metavar='S',
                    help='source dataset')
parser.add_argument('--tgt', type=str, default='t', metavar='S',
                    help='target dataset')
parser.add_argument('--print', type=int, default=100,
                    help='print interval')
parser.add_argument('--test', type=int, default=100,
                    help='test interval')
parser.add_argument('--lamda', type=float, default=0.00005,
                    help='nmf loss hyperparameter')
parser.add_argument('--lr', type=float, default=0.1,
                        help='init learning rate for fine-tune')
parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
parser.add_argument('--lam_delta', type=float, default=1.0,
                        help='hyperparameter for delta dictionary')
args = parser.parse_args()


torch.manual_seed(args.seed)
np.random.seed(args.seed)
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

os.chdir(r'/home/entitees/labo-litis/users/dhainmoh/data/mpi3d')

rc="realistic.txt"
rl="real.txt"
t="toy.txt"

rc_t="realistic_test.txt"
rl_t="real_test.txt"
t_t="toy_test.txt"

if args.src =='rl':
    source_path = rl
elif args.src =='rc':
    source_path = rc
elif args.src =='t':
    source_path = t


if args.tgt =='rl':
    target_path = rl
elif args.tgt =='rc':
    target_path = rc
elif args.tgt =='t':
    target_path = t

if args.tgt =='rl':
    target_path_t = rl_t
elif args.tgt =='rc':
    target_path_t = rc_t
elif args.tgt =='t':
    target_path_t = t_t
dsets = {"train": ImageList(open(source_path).readlines(), transform=data_transforms["train"]),
         "val": ImageList(open(target_path).readlines(),transform=data_transforms["val"]),
         "test": ImageList(open(target_path_t).readlines(),transform=data_transforms["test"])}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],
                                               shuffle=True, num_workers=0)
                for x in ['train', 'val']}
dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=batch_size["test"],
                                                   shuffle=False, num_workers=0)

dset_sizes = {x: len(dsets[x]) for x in ['train', 'val','test']}
# device = torch.device('cuda')

def match_nmf_v3(Feature_s, Feature_t):
    _,res1,b_s,A_s,R_s=ls.nmf_mu(Feature_s.T,18,0,0,100)
    loss,res2,b_t,A_t,R_t=ls.nmf_mu(Feature_t.T,18,0,0,100)
    # print(torch.norm(b_s,p=2))
    # plt.plot(loss)
    res_s,_,aa,_=ls.sparse(b_s,b_t,0,0,100)
    res_t,_,_,loss=ls.sparse(b_t,b_s,0,0,100)
    # plt.plot(loss)
    # print(aa)
    return torch.norm(res_s,p='fro')+torch.norm(res_t,p='fro')
  
def match_nmf_v5(Feature_s, Feature_t):
  b_s,A_s,error_s,loss_s=ls.dictionary_learning(50,Feature_s.t(),rank=18,lambda_sp=0.9,lambda_reg=1,lamda = 0.1)
  coeffs1,loss= ls.ista(Feature_t,b_s,alpha=0.4,maxiter=50)
  # plt.plot(loss)
  res_s=Feature_t.T-b_s.mm(coeffs1.T)
  delta=res_s.mm(coeffs1.mm(torch.linalg.pinv(args.lam_delta*torch.ones(18,18).cuda()+coeffs1.T.mm(coeffs1))))
  return torch.norm(delta,p='fro')

def Regression_test(loader, model):
    MSE = [0, 0, 0]
    MAE = [0, 0, 0]
    number = 0
    with torch.no_grad():
        for (imgs, labels) in loader['test']:
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels1 = labels[:, 0]
            labels2 = labels[:, 1]
            labels1 = labels1.unsqueeze(1)
            labels2 = labels2.unsqueeze(1)
            labels = torch.cat((labels1, labels2), dim=1)
            labels = labels.float() / 39
            pred = model(imgs)
            MSE[0] += torch.nn.MSELoss(reduction='sum')(pred[:, 0], labels[:, 0])
            MAE[0] += torch.nn.L1Loss(reduction='sum')(pred[:, 0], labels[:, 0])
            MSE[1] += torch.nn.MSELoss(reduction='sum')(pred[:, 1], labels[:, 1])
            MAE[1] += torch.nn.L1Loss(reduction='sum')(pred[:, 1], labels[:, 1])
            MSE[2] += torch.nn.MSELoss(reduction='sum')(pred, labels)
            MAE[2] += torch.nn.L1Loss(reduction='sum')(pred, labels)
            number += imgs.size(0)
    for j in range(3):
        MSE[j] = MSE[j] / number
        MAE[j] = MAE[j] / number
    print("\tMSE : {0},{1}\n".format(MSE[0], MSE[1]))
    print("\tMAE : {0},{1}\n".format(MAE[0], MAE[1]))
    print("\tMSEall : {0}\n".format(MSE[2]))
    print("\tMAEall : {0}\n".format(MAE[2]))
    return MAE[2]

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
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        self.classifier_layer = nn.Sequential(self.classifier_layer,  nn.Sigmoid())
        self.predict_layer = nn.Sequential(self.model_fc,self.classifier_layer)
    def forward(self,x):
        feature = self.model_fc(x)
        outC= self.classifier_layer(feature)
        return(outC,feature)


test_graph_nmf=[]
set_seed()
name='nmf'
domain_graph=[]
regression_graph=[]
Model_R = Model_Regression()
Model_R = Model_R.to(device)
Model_R.train(True)
criterion = {"regressor": nn.MSELoss()}
optimizer_dict = [{"params": filter(lambda p: p.requires_grad, Model_R.model_fc.parameters()), "lr": 0.1},
                  {"params": filter(lambda p: p.requires_grad, Model_R.classifier_layer.parameters()), "lr": 1}]
optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
train_cross_loss = train_nmf_loss = train_total_loss = 0.0
len_source = len(dset_loaders["train"]) - 1
len_target = len(dset_loaders["val"]) - 1
param_lr = []
iter_source = iter(dset_loaders["train"])
iter_target = iter(dset_loaders["val"])
for param_group in optimizer.param_groups:
    param_lr.append(param_group["lr"])
test_interval = args.test
print_interval=args.print
num_iter = num_iter = 1*len_source
test_init=np.inf
writer = SummaryWriter(log_dir=os.path.join(r'/home/scr/person/invite/dhainmoh/labo-litis/dhainmoh/scripts/nmf-adapt',args.src+args.tgt))
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
    labels1 = labels_source[:,0]
    labels2 = labels_source[:,1]
    labels1 = labels1.unsqueeze(1)
    labels2 = labels2.unsqueeze(1)
    labels_source = torch.cat((labels1,labels2),dim=1)
    labels_source = labels_source.float()/39
    inputs_target, labels_target = data_target
    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    inputs = inputs.to(device)
    labels = labels_source.to(device)
    inputs_s = inputs.narrow(0, 0, batch_size["train"])
    inputs_t = inputs.narrow(0, batch_size["train"], batch_size["train"])
    outC_s, feature_s = Model_R(inputs_s)
    outC_t, feature_t = Model_R(inputs_t)
    classifier_loss = criterion["regressor"](outC_s, labels)
    # beta = 0.001*(1 + 0.0001 * iter_num) ** (-0.75)
    # print(torch.norm(feature_s,p=2),torch.norm(feature_t,p=2))
#     if iter_num<3000:
#       total_loss = classifier_loss
#     else:
    nmf_loss= match_nmf_v5(feature_s,feature_t)
    total_loss = classifier_loss + args.lamda*nmf_loss
    train_nmf_loss += nmf_loss.item()
    total_loss.backward()
    # print(Model_R.model_fc.layer4[1].bn2.weight.grad)
    # torch.nn.utils.clip_grad_norm_(Model_R.parameters(), max_norm=3, norm_type=2)
    # for p in Model_R.parameters():
    #     p.register_hook(lambda grad: print(torch.norm(grad,p=2)))
    optimizer.step()
    train_cross_loss += classifier_loss.item()
#     train_nmf_loss += nmf_loss.item()
    train_total_loss += total_loss.item()
    # end_=time.time()
    # print(end_-start)
    # if iter_num==num_iter:
    #   torch.save(Model_R.state_dict(), os.path.join('/content/drive/MyDrive/Colab Notebooks/domain adaptation','regressor_epoch_scream'))
    if (iter_num % print_interval) == 0:
        print("Iter {:05d}, Average Cross Entropy Loss: {:.4f}; Average NMF Loss: {:.4f};  Average Training Loss: {:.4f};  LR:{:.6f}".format(
            iter_num, train_cross_loss / float(test_interval), train_nmf_loss / float(test_interval),
            train_total_loss / float(test_interval),optimizer.param_groups[0]['lr']))
        domain_graph.append(train_nmf_loss / float(test_interval))
        regression_graph.append( train_cross_loss / float(test_interval))
#             np.save(os.path.join('/content/drive/MyDrive/Colab Notebooks/nmf domain adaptation','domain_loss.npy'),np.array(domain_graph))
#             np.save(os.path.join('/content/drive/MyDrive/Colab Notebooks/nmf domain adaptation','regression_loss.npy'),np.array(regression_graph))
        train_cross_loss = train_nmf_loss = train_total_loss  = 0.0
        # beta=beta*0.9
    if (iter_num % test_interval) == 0:
        Model_R.eval()
        test_loss=Regression_test(dset_loaders, Model_R.predict_layer)
        writer.add_scalar("Loss/Test Loss", test_loss , iter_num)
        if test_loss<test_init:
            test_init=test_loss
            print('Saving')
#                 torch.save(Model_R.state_dict(), os.path.join('/content/drive/MyDrive/Colab Notebooks/nmf domain adaptation','nmf2'))
#                 torch.save(optimizer.state_dict(),os.path.join('/content/drive/MyDrive/Colab Notebooks/nmf domain adaptation','optimizer2'))
#                 np.save(os.path.join('/content/drive/MyDrive/Colab Notebooks/nmf domain adaptation','iter2.npy'),np.array(iter_num))
        test_graph_nmf.append(test_loss)
        test_=[s.item() for s in test_graph_nmf]
#                 np.save(os.path.join('/content/drive/MyDrive/Colab Notebooks/nmf domain adaptation','test_loss.npy'),np.array(test_))
