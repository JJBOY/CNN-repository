import Inception
import torchvision
import torch
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from utils import *

device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
weight_decay=0.0001
batch_size=128
num_epochs=200
save_path='./record/'

def train_model(model,dataloaders,criterion,optimizer,num_epochs):
    
    since=time.time()
    val_acc_history=[]
    train_acc_history=[]
    best_model_wts=copy.deepcopy(model.state_dict())
    best_acc=0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1,num_epochs))
        print('-'*10)
        if epoch==100:
            optimizer.param_groups[0]['lr']/=10
        if epoch==150:
            optimizer.param_groups[0]['lr']/=10

        for phase in ['train','val']:
            if phase=='train':
                model.train()
            else:
                model.eval()

            running_loss=0.0
            running_corrects=0.0

            for inputs,labels in dataloaders[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs,aux_outputs1,aux_outputs2=model(inputs)
                    loss1=criterion(outputs,labels)
                    loss2=criterion(aux_outputs1,labels)
                    loss3=criterion(aux_outputs2,labels)
                    loss=loss1+0.3*loss2+0.3*loss3
                    _,preds=torch.max(outputs,1)

                    if(phase=='train'):
                        loss.backward()
                        optimizer.step()

                running_loss+=loss.item()*inputs.size(0)
                running_corrects+=torch.sum(preds==labels.data).item()

            epoch_loss=running_loss/len(dataloaders[phase].dataset)
            epoch_acc=running_corrects.double()/len(dataloaders[phase].dataset)

            info = {'Epoch': [epoch+1],
                    'Loss': [epoch_loss],
                    'Acc': [epoch_acc],
                }
            record_info(info, 'record/'+phase+'.csv')

            if phase=='val' and epoch_acc>best_acc:
                best_acc=epoch_acc
                best_model_wts=copy.deepcopy(model.state_dict())
            if phase=='val':
                val_acc_history.append(epoch_acc)
            else:
                train_acc_history.append(epoch_acc)
        print()

    time_elapsed=time.time()-since
    print('Training complete in {:.0f}m {:0f}s'.format(time_elapsed//60,time_elapsed%60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model,val_acc_history,train_acc_history,best_acc

def savehis(val_hist,train_hist,path,name,best_acc,lr=None):
    plt.title("Accuracy vs. Number of Training Epochs")
    plt.xlabel('Training Epochs')
    plt.ylabel('Accuracy')
    plt.plot(range(1,num_epochs+1),val_hist,label=name+' val:'+'%.4f'%(best_acc))
    plt.plot(range(1,num_epochs+1),train_hist,label=name+' training')
    plt.ylim(0,1.0)
    plt.xticks(np.arange(0,num_epochs+1,20))
    plt.legend()
    plt.savefig(path)   


def main():
    transform=torchvision.transforms.Compose([
            torchvision.transforms.Pad(4),
            torchvision.transforms.RandomCrop((32,32)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ])
    train_dataset=torchvision.datasets.CIFAR10('../cifar10-data', train=True, transform=transform)
    test_dataset=torchvision.datasets.CIFAR10('../cifar10-data', train=False, 
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]))
    dataloaders_dict={
        'train':torch.utils.data.DataLoader(
            train_dataset,batch_size=batch_size,shuffle=True,num_workers=4
            ) ,
        'val':torch.utils.data.DataLoader(
            test_dataset,batch_size=batch_size,shuffle=True,num_workers=4
            ) 
        }

   
    all_net={
            'BN-Inception 1e-1':Inception.InceptionNet(num_classes=10,batch_norm=True),
            'BN-Inception 1e-2':Inception.InceptionNet(num_classes=10,batch_norm=True),
            'BN-Inception 1e-3':Inception.InceptionNet(num_classes=10,batch_norm=True)
            }
    lr=0.1
    for name ,net in all_net.items():
        net=net.to(device)
        optimizer=torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9,weight_decay=weight_decay)
        criterion=torch.nn.CrossEntropyLoss()
        net,val_hist,train_hist,best_acc=train_model(net,dataloaders_dict,criterion,optimizer,num_epochs)
        #torch.save(net.state_dict(), save_path+name+'.pth')
        val_hist=[h.cpu().numpy() for h in val_hist]
        train_hist=[h.cpu().numpy() for h in train_hist]
        savehis(val_hist,train_hist,save_path+name+'.png',name,best_acc)
        lr/=10

    
if __name__ == '__main__':
    main()
