import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class InputsDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs, _ = self.dataset[idx]
        return inputs



def load_mnist_data():

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = InputsDataset(
        torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform))

    test_dataset = InputsDataset(
        torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform))

    return train_dataset, test_dataset
