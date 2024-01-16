import torch
from torchvision import datasets,  transforms
    

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main(dataset):
    BATCH_SIZE = 100
    TEST_BATCH_SIZE = 100
    
    kwargs = {'num_workers': 6,  'pin_memory': True} if torch.cuda.is_available() else {}

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]
                                    ),
                                   ])
    if dataset == "cifar10":
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./cifar10_data', 
                                                                    train=True, 
                                                                    download=True, 
                                                                    transform=transform),  
                                                                    batch_size=BATCH_SIZE, 
                                                                    shuffle=True, 
                                                                    **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./cifar10_data',  
                                                                train=False,  
                                                                download=True, 
                                                                transform=transform),  
                                                                batch_size=TEST_BATCH_SIZE, 
                                                                shuffle=True, 
                                                                **kwargs)
    elif dataset == "cifar100":
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR100('./cifar100_data', 
                                                                    train=True, 
                                                                    download=True, 
                                                                    transform=transform),  
                                                                    batch_size=BATCH_SIZE, 
                                                                    shuffle=True, 
                                                                    **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.CIFAR100('./cifar100_data',  
                                                                train=False,  
                                                                download=True, 
                                                                transform=transform),  
                                                                batch_size=TEST_BATCH_SIZE, 
                                                                shuffle=True, 
                                                                **kwargs)
    return train_loader, test_loader


if __name__ == '__main__':
    main()