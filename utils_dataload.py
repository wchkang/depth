import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

#Prepares training dataset
#dataset: String, name of dataset
#root: String, path of dataset
def get_traindata(dataset, root, download=False, shuffle=True, batch_size=128, num_workers=8):
        
    if dataset=='CIFAR10':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        data = torchvision.datasets.__dict__[dataset](root=root, train=True, download=download, transform=transform)
        trainloader = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)
        
    elif dataset=='CIFAR100':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        
        data = torchvision.datasets.__dict__[dataset](root=root, train=True, download=download, transform=transform)
        trainloader = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)
        
    elif dataset=='ILSVRC2012':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        
        data = datasets.ImageFolder(
        root+"/train/",
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
        
        trainloader = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)
        
    else:
        print('Not implemented')
        trainloader = None
    
    return trainloader

#Prepares test dataset
#dataset: String, name of dataset
#root: String, path of dataset
def get_testdata(dataset, root, download=False, shuffle=False, batch_size=128, num_workers=8):
    
    if dataset=='CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        data = torchvision.datasets.__dict__[dataset](root=root, train=False, download=download, transform=transform)
        testloader = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)
    
    elif dataset=='CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        data = torchvision.datasets.__dict__[dataset](root=root, train=False, download=download, transform=transform)
        testloader = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)
    
    elif dataset=='ILSVRC2012':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        
        testloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root+"/val/", transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=True)
        
    else:
        print('Not implemented')
        testloader = None
        
    return testloader
