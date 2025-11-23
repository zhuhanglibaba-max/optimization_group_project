import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter
import torch
import random

model_name = 'VGG16'

def ploter_ax(ax:Axes, sgd_y, nsgd_y, y_label, title):
    x_range = list(range(len(sgd_y)))
    label_size = 20
    tick_size = 18
    ax.plot(x_range, sgd_y, color='orange', linewidth=2.5, marker='s', markersize=2.5, markerfacecolor='none', label='SGD', markevery=20)
    ax.plot(x_range, nsgd_y, color='blue', linewidth=2.5, marker='^', markersize=2.5, markerfacecolor='none', label='NoisySGD', markevery=20)
    ax.set_ylabel(ylabel=y_label, fontsize=label_size)
    ax.set_xlabel('Epochs', fontsize=label_size)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid(linestyle='dashed')
    ax.legend(shadow=True, fontsize=tick_size)
    ax.set_facecolor('whitesmoke')
    ax.set_title(title, fontsize=label_size)

item_name = ['train_loss', 'test_loss', 'train_acc', 'test_acc', 'loss_gap', 'acc_gap', 'gradient_norm']

for item in item_name:
    fig, ax = plt.subplots(1,3, figsize = (14,4))

    dataset_name = 'SVHN'
    root = 'CourseWork/{}/{}/'.format(dataset_name, model_name)

    sgd_data = torch.load(root+'sgd.pth')
    nsgd_data = torch.load(root+'nsgd.pth')

    sgd_train_loss = sgd_data[item]
    nsgd_train_loss = nsgd_data[item]

    ploter_ax(ax[0], sgd_train_loss, nsgd_train_loss, item, dataset_name)

    dataset_name = 'CIFAR10'
    root = 'CourseWork/{}/{}/'.format(dataset_name, model_name)

    sgd_data = torch.load(root+'sgd.pth', weights_only=True)
    nsgd_data = torch.load(root+'nsgd.pth', weights_only=True)

    sgd_train_loss = sgd_data[item]
    nsgd_train_loss = nsgd_data[item]

    ploter_ax(ax[1], sgd_train_loss, nsgd_train_loss, item, dataset_name)

    dataset_name = 'CIFAR100'
    root = 'CourseWork/{}/{}/'.format(dataset_name, model_name)

    sgd_data = torch.load(root+'sgd.pth', weights_only=True)
    nsgd_data = torch.load(root+'nsgd.pth', weights_only=True)

    sgd_train_loss = sgd_data[item]
    nsgd_train_loss = nsgd_data[item]

    ploter_ax(ax[2], sgd_train_loss, nsgd_train_loss, item, dataset_name)

    plt.tight_layout()
    plt.savefig('Coursework_{}.png'.format(item))
    plt.close()



