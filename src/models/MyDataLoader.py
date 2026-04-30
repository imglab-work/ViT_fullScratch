from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MNISTDataLoader:
    def __init__(self, batch_size=64, dataset="MNIST", root='./data'):
        # 1. 前処理をコンストラクタで定義
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # 2. データセットの準備
        if dataset == "MNIST":
            train = datasets.MNIST(root=root, train=True, download=True, transform=self.transform)
            test = datasets.MNIST(root=root, train=False, download=True, transform=self.transform)
        elif dataset == "FashionMNIST":
            train = datasets.FashionMNIST(root=root, train=True, download=True, transform=self.transform)
            test = datasets.FashionMNIST(root=root, train=False, download=True, transform=self.transform)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        self.train_dataset = train
        self.test_dataset = test
        
        # 3. 効率化のため、DataLoaderをあらかじめ作成して保持しておく
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def get_train(self):
        return self.train_loader

    def get_test(self):
        return self.test_loader