from model import *
from torchvision import transforms
import os
from picker import *
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
from torch.optim import optimizer
from torch.optim.lr_scheduler import MultiStepLR

class NSGD(optimizer.Optimizer):
    '''
        Implements the SGLD
    '''
    def __init__(self, params, lr=1e-1, momentum=0., weight_decay=0.0,eps=1e-6, noise=0.3, gamma=0.):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= noise:
            raise ValueError("Invalid noise value: {}".format(noise))
        if not 0.0 <= gamma:
            raise ValueError("Invalid noise value: {}".format(gamma))
        
        
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps, noise=noise, gamma=gamma)
        super(NSGD, self).__init__(params, defaults)
            
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('SGLD does not support sparse gradients')
                state = self.state[p]

                if weight_decay != 0:
                    grad.add_(weight_decay, p.data)

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    
                state['step'] += 1
                
                std = torch.tensor(math.sqrt(2./group['lr'])*group['noise'])
                nr = torch.normal(mean=0, std = std)
                updt = grad.add(nr)
                p.data.add_(-group['lr'],updt)
                
        return loss

model_list =['ResNet18', 'ResNet50', 'ResNet101','ResNet152']
dataset_list = ['CIFAR10', 'CIFAR100', 'SVHN']
optimizer_list = ['SGD', 'BSGD','LRSGD', 'MSGD', 'WDSGD']

dataset_name = 'SVHN'
num_classes = 10
model_name = 'VGG16'
root = 'CourseWork/{}/{}/'.format(dataset_name, model_name)
device = 'cuda:5'

if not os.path.exists(root):
    os.makedirs(root)

data_tf = {
        'train':transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32,4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ]),
        'test':transforms.Compose([
            transforms.ToTensor(),
        ])
        }

model = model_picker(model_name, num_classes)
opt = NSGD(model.parameters(), 0.01, 0.9, 5e-4, noise=0.0001)
stl = MultiStepLR(opt, milestones=[100,150], gamma=0.1)
trainset, testset = dataset_picker(dataset_name, 
                                   '/home/yulin/data',
                                   data_tf['train'],
                                   data_tf['test'])

trainloader = DataLoader(trainset, batch_size=256, shuffle=True)
testloader = DataLoader(testset, batch_size=1000, shuffle=False)
evalloader = DataLoader(trainset, batch_size=1000, shuffle=False)


train_loss = []
test_loss = []
loss_gap = []
gradient_norm = []

loss_func = nn.CrossEntropyLoss()

model.to(device)

train_acc = []
test_acc = []
acc_gap = []

for i in range(200):
    print('Begin ***************{}-th epoch training***************'.format(i+1))
    model.train()

    epoch_train_loss = []
    epoch_gradient_norm = []
    epoch_train_acc = []

    for (data, label) in tqdm(trainloader, desc='training', total=len(trainloader), colour='red'):
        data, label = data.to(device), label.to(device)
        output = model(data)
        _, pred = torch.max(output, dim=1)
        correct = (pred == label).sum().item()
        epoch_train_acc.append(correct)

        loss = loss_func(output, label)

        B = data.shape[0]
        epoch_train_loss.append(B*loss.item())

        opt.zero_grad()
        loss.backward()
        
        gradient = 0.
        for p in model.parameters():
            gradient += p.grad.data.norm().pow(2).item()
        gradient = B*math.sqrt(gradient)
        epoch_gradient_norm.append(gradient)

        opt.step()

    stl.step()

    train_loss.append(sum(epoch_train_loss)/len(trainset))
    gradient_norm.append(sum(epoch_gradient_norm)/len(trainset))
    train_acc.append(sum(epoch_train_acc)/len(trainset))

    print('Begin ***************{}-th epoch testing***************'.format(i+1))
    model.eval()
    with torch.no_grad():
        epoch_test_loss = []
        epoch_test_acc = []
        for (data, label) in tqdm(testloader, desc='evaling', total=len(testloader), colour='blue'):
            data, label = data.to(device), label.to(device)
            output = model(data)
            _, pred = torch.max(output, dim=1)
            correct = (pred == label).sum().item()
            loss = loss_func(output, label)

            B = data.shape[0]
            epoch_test_loss.append(B*loss.item())
            epoch_test_acc.append(correct)

    test_loss.append(sum(epoch_test_loss)/len(testset))
    test_acc.append(sum(epoch_test_acc)/len(testset))

    loss_gap.append(test_loss[-1]-train_loss[-1])
    acc_gap.append(train_acc[-1] - test_acc[-1])

    print(train_loss[-1], test_loss[-1])
    print(train_acc[-1], test_acc[-1])
    print(loss_gap[-1], acc_gap[-1])
    print(gradient_norm[-1])

    load_dict = {
        'train_loss':train_loss,
        'test_loss':test_loss,
        'train_acc':train_acc,
        'test_acc':test_acc,
        'loss_gap':loss_gap,
        'acc_gap':acc_gap,
        'gradient_norm':gradient_norm
    }

    torch.save(load_dict, root+'nsgd.pth')