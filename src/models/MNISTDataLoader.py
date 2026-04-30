from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MNISTDataLoader:
    def __init__(self, batch_size=64, root='./data'):
        # 1. 前処理をコンストラクタで定義
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # 2. データセットの準備
        self.train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=self.transform)
        
        # 3. 効率化のため、DataLoaderをあらかじめ作成して保持しておく
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def get_train(self):
        return self.train_loader

    def get_test(self):
        return self.test_loader