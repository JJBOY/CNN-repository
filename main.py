from model import DenseNet
import torchvision
import torch
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from model.utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print(device)
weight_decay = 0.0005
batch_size = 64
num_epochs = 60
save_path = './record/'

import os

if os.path.exists(save_path) == False:
    os.mkdir(save_path)


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()
    val_acc_history = []
    train_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        if epoch == 40:
            optimizer.param_groups[0]['lr'] /= 10
        if epoch == 53:
            optimizer.param_groups[0]['lr'] /= 10

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if (phase == 'train'):
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            info = {'Epoch': [epoch + 1],
                    'Loss': [epoch_loss],
                    'Acc': [epoch_acc],
                    }
            record_info(info, 'record/' + phase + '.csv')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            else:
                train_acc_history.append(epoch_acc)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, best_acc


def savehis(val_hist, train_hist, path, name, best_acc, lr=None):
    plt.title("Accuracy vs. Number of Training Epochs")
    plt.xlabel('Training Epochs')
    plt.ylabel('Accuracy')
    plt.plot(range(1, num_epochs + 1), val_hist, label=name + ' val:' + '%.4f' % (best_acc))
    plt.plot(range(1, num_epochs + 1), train_hist, label=name + ' training')
    plt.ylim(0, 1.0)
    plt.xticks(np.arange(0, num_epochs + 1, 20))
    plt.legend()
    plt.savefig(path)


def main():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Pad(4),
        torchvision.transforms.RandomCrop((32, 32)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.CIFAR10('/home/qx/project/data/cifar10-data', train=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10('/home/qx/project/data/cifar10-data', train=False,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                ]))
    dataloaders_dict = {
        'train': torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        ),
        'val': torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
    }

    all_net = {
        #'DenseNet29': DenseNet.densenet29(),
        #'DenseNet45': DenseNet.densenet45(),
        'DenseNet85': DenseNet.densenet85()
    }
    lr = 0.1
    for name, net in all_net.items():
        net = net.to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        net, val_hist, train_hist, best_acc = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs)
        # torch.save(net.state_dict(), save_path+name+'.pth')
        val_hist = [h for h in val_hist]
        train_hist = [h for h in train_hist]
        savehis(val_hist, train_hist, save_path + name + '.png', name, best_acc)


if __name__ == '__main__':
    main()
