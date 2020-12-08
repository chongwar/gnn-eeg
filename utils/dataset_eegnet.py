from torchvision import transforms
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, x_train, x_test, y_train, y_test, train=True):
        super(MyDataset, self).__init__()
        self.transforms = transforms.ToTensor()
        self.x = x_train if train else x_test
        self.y = y_train if train else y_test
        
    def __getitem__(self, index):
        x = self.x[index, ...]
        y = self.y[index, ...]
        return x, y
    
    def __len__(self):
        return len(self.x)
    