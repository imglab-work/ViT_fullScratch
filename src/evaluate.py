from ViT import ViT
import torch
from MNISTDataLoader import MNISTDataLoader
import matplotlib.pyplot as plt

# 新しいモデルのインスタンスを作る（形を同じにする必要があります）
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

# 保存した重みを読み込む
model.load_state_dict(torch.load("vit_mnist_model.pth"))

data_manager = MNISTDataLoader(batch_size=64)
train_loader = data_manager.get_train()
test_loader = data_manager.get_test()


def evaluate():
    # 推論モードにする（これ大事！）
    model.eval()
    # 学習完了後、こんなコードで正解率を測れます
    model.eval() # 評価モード
    correct = 0
    with torch.no_grad(): # 勾配計算をオフにしてメモリ節約
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / len(data_manager.test_dataset)}%")



def predict():
    # テストデータから1バッチ（64枚）取得
    images, labels = next(iter(test_loader))

    # 最初の1枚だけで予測
    model.eval()
    with torch.no_grad():
        output = model(images[0:1]) # 最初の1枚だけスライス
        _, predicted = torch.max(output, 1)

    print(f"AIの予測: {predicted.item()}")
    print(f"正解ラベル: {labels[0].item()}")

    # 画像を表示（WSL環境などの場合は、保存して確認するのがおすすめ）
    # plt.imshow(images[0].reshape(28, 28), cmap='gray')
    # plt.show()
predict()