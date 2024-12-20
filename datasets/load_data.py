import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_mnist(batch_size=64, download=True):
    # Define transformation to convert grayscale to RGB (3 channels)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ])
    
    # Load the training and testing datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=download, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader