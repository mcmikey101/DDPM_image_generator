import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

class NumDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda t: (t * 2) - 1)
                                ]) 
        self.data = torchvision.datasets.MNIST(root=".", download=True, transform=self.transform)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]