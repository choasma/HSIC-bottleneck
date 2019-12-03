from .. import *


def get_dataset_from_code(code, batch_size):
    """ interface to get function object
    Args:
        code(str): specific data type
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    dataset_root = "./assets/data"
    if code == 'mnist':
        train_loader, test_loader = get_mnist_data(batch_size=batch_size,
            data_folder_path=os.path.join(dataset_root, 'mnist-data'))
    elif code == 'cifar10':
        train_loader, test_loader = get_cifar10_data(batch_size=batch_size,
            data_folder_path=os.path.join(dataset_root, 'cifar10-data'))
    elif code == 'fmnist':
        train_loader, test_loader = get_fasionmnist_data(batch_size=batch_size,
            data_folder_path=os.path.join(dataset_root, 'fasionmnist-data'))
    else:
        raise ValueError("Unknown data type : [{}] Impulse Exists".format(data_name))

    return train_loader, test_loader


def get_fasionmnist_data(data_folder_path, batch_size=64):
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                   #transforms.Normalize((0.2860,), (0.3530,)),
                                 ])
    # Download and load the training data
    trainset = datasets.FashionMNIST(data_folder_path, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.FashionMNIST(data_folder_path, download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

def get_mnist_data(data_folder_path, batch_size=64):
    """ mnist data
    Args:
        train_batch_size(int): training batch size 
        test_batch_size(int): test batch size
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    train_data = datasets.MNIST(data_folder_path, train=True,  download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(), 
            #transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

    test_data  = datasets.MNIST(data_folder_path, train=False, download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(), 
            #transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=batch_size, shuffle=False, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_data,  
        batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader

def get_cifar10_data(data_folder_path, batch_size=64):
    """ cifar10 data
    Args:
        train_batch_size(int): training batch size 
        test_batch_size(int): test batch size
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    transform_train = transforms.Compose([

        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615)),

    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615)),
    ])

    train_data = datasets.CIFAR10(data_folder_path, train=True, 
        download=True, transform=transform_train)
    test_data  = datasets.CIFAR10(data_folder_path, train=False, 
        download=True, transform=transform_test) 

    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=batch_size, shuffle=False, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_data, 
        batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader
