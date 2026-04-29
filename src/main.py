from ViT import ViT
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 前処理の定義
# 画像をPyTorchが扱えるTensor形式に変換し、数値を正規化します
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # 平均0.5, 標準偏差0.5で正規化
])

# 2. MNISTデータセットのダウンロードと読み込み
# 学習用データ
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# テスト用（評価用）データ
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 3. DataLoader（バッチごとにデータを取り出す役職）の設定
# batch_size=64なら、一度に64枚の画像をまとめてモデルに渡します
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"学習データ数: {len(train_dataset)}枚")



# モデルのインスタンス化
model = ViT(
    image_size=224, 
    patch_size=16, 
    n_classes=10, 
    dim=128, 
    depth=6, 
    n_heads=8, 
    mlp_dim=256
)





# 1. モデル、損失関数、最適化手法を準備
device = torch.device("cpu") # CPUを明示的に指定
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4) # lrは学習率



# 1バッチ分だけ取り出してみる
images, labels = next(iter(train_loader))

# モデルに通してみる
outputs = model(images)

print(f"入力の形: {images.shape}")  # [64, 1, 28, 28]
print(f"出力の形: {outputs.shape}") # [64, 10]


# 2. 学習ループ
epochs = 5 # 全データを何回繰り返して学習するか
for epoch in range(epochs):
    model.train() # 学習モードに設定
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        # データをCPUへ
        images, labels = images.to(device), labels.to(device)
        
        # --- ここからが1ステップの基本 ---
        # (1) 勾配をゼロにリセット（前回の計算結果が残らないように）
        optimizer.zero_grad()
        
        # (2) 予測（順伝播）
        outputs = model(images)
        
        # (3) 誤差の計算
        loss = criterion(outputs, labels)
        
        # (4) 誤差逆伝播（どの重みを直すべきか計算）
        loss.backward()
        
        # (5) 重みの更新
        optimizer.step()
        # --------------------------------
        
        running_loss += loss.item()
        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print("学習完了！")