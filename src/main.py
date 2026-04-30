from ViT import ViT
import torch
import torch.optim as optim
import torch.nn as nn


from models.MNISTDataLoader import MNISTDataLoader


data_manager = MNISTDataLoader(batch_size=64)
train_loader = data_manager.get_train()
#test_loader = data_manager.get_test()

# モデルのインスタンス化
# 修正後のインスタンス化
model = ViT(
    image_size=28,      # 224 から 28 へ変更
    patch_size=7,       # 16 から 7 へ変更（28を割り切れる数にする）
    n_classes=10, 
    channels=1,         # MNISTは白黒なので 1 を追加（クラス引数にある場合）
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

# モデルの重みを保存する
torch.save(model.state_dict(), "vit_mnist_model.pth")
print("モデルを保存しました：vit_mnist_model.pth")