from model import *
from torchvision import transforms
import os
from picker import *
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

model_list =['ResNet18', 'ResNet50', 'ResNet101','ResNet152']
dataset_list = ['CIFAR10', 'CIFAR100', 'SVHN']
optimizer_list = ['SGD', 'BSGD','LRSGD', 'MSGD', 'WDSGD']

optimizer = 'SGD'
dataset_name = 'SVHN'
num_classes = 10
model_name = 'VGG16'
root = 'CourseWork/{}/{}/'.format(dataset_name, model_name)
device = 'cuda:2'

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
opt = optimizer_picker(model.parameters(), 0.01, 0.9, 5e-4)
stl = MultiStepLR(opt, [100, 150], gamma=0.1)

trainset, testset = dataset_picker(dataset_name, 
                                   '/home/yulin/data',
                                   data_tf['train'],
                                   data_tf['test'])

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=500, shuffle=False)
evalloader = DataLoader(trainset, batch_size=500, shuffle=False)


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


    for (data, label) in tqdm(trainloader, desc='evaling', total=len(trainloader), colour='red'):
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss = loss_func(output, label)
        opt.zero_grad()
        loss.backward()
        opt.step()

    stl.step()
    epoch_train_loss = []
    epoch_gradient_norm = []
    epoch_train_acc = []

    print('Begin ***************{}-th epoch evaling***************'.format(i+1))
    model.eval()
    for (data, label) in tqdm(evalloader, desc='evaling', total=len(evalloader), colour='blue'):
        data, label = data.to(device), label.to(device)
        output = model(data)
        _, pred = torch.max(output, dim=1)
        correct = (pred == label).sum().item()
        epoch_train_acc.append(correct)

        loss = loss_func(output, label)

        opt.zero_grad()
        loss.backward()

        B = data.shape[0]
        epoch_train_loss.append(B*loss.item())

        gradient = 0.
        for p in model.parameters():
            gradient += p.grad.data.norm().pow(2).item()
        gradient = B*math.sqrt(gradient)
        epoch_gradient_norm.append(gradient)


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

    torch.save(load_dict, root+'sgd.pth')