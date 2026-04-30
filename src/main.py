import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time
from ViT import ViT
from models.MyDataLoader import MNISTDataLoader
# 前回の評価クラスを再利用（別ファイルに保存している前提、もしくは同ファイル内に定義）
from evaluate import ViTEvaluator
import shutil 
from config import Config

def train_and_evaluate():
    # --- 1. 基本設定とフォルダ準備 ---
    BATCH_SIZE = Config.BATCH_SIZE
    EPOCHS = Config.EPOCHS
    LR = Config.LR
    
    # 実行時の時刻でフォルダ名を作成 (例: results_20260430_1430)
    timestamp = time.strftime("%Y%m%d_%H%M")
    save_dir = f"results/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    conf_path = os.path.join(save_dir, "config_backup.py")
    shutil.copy("src/config.py", conf_path)
    print(f"📂 Results will be saved in: {save_dir}")

    # --- 2. モデル・データ準備 ---
    data_manager = MNISTDataLoader(batch_size=BATCH_SIZE, dataset=Config.DATASET)
    train_loader = data_manager.get_train()
    test_loader = data_manager.get_test()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT(
        image_size=Config.IMAGE_SIZE,
        patch_size=Config.PATCH_SIZE,
        n_classes=Config.N_CLASSES,
        channels=Config.CHANNELS,
        dim=Config.DIM,
        depth=Config.DEPTH,
        n_heads=Config.N_HEADS,
        mlp_dim=Config.MLP_DIM
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # --- 3. 学習ループ (Lossの記録付き) ---
    history = []
    print("🚀 Starting training...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, attentions = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                avg_loss = running_loss / 100
                print(f"[{epoch + 1}, {i + 1}] loss: {avg_loss:.3f}")
                history.append(avg_loss) # 100ステップごとの平均Lossを記録
                running_loss = 0.0

    # --- 4. 成果物の保存 (重み & 学習曲線) ---
    # 重みの保存
    weight_path = os.path.join(save_dir, "vit_mnist_model.pth")
    torch.save(model.state_dict(), weight_path)
    print(f"💾 Model weights saved to: {weight_path}")

    # 学習曲線の保存
    plt.figure(figsize=(8, 5))
    plt.plot(history, label="Training Loss")
    plt.xlabel("Steps (x100)")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    curve_path = os.path.join(save_dir, "learning_curve.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"📈 Learning curve saved to: {curve_path}")

    # --- 5. そのまま評価フェーズへ ---
    print("\n🧐 Starting evaluation...")
    evaluator = ViTEvaluator(model_path=weight_path, conf_path=conf_path)
    evaluator.evaluate(test_loader, save_dir=save_dir)

if __name__ == "__main__":
    train_and_evaluate()