import torch
import matplotlib.pyplot as plt
import os
from ViT import ViT
from MNISTDataLoader import MNISTDataLoader

def run_inference():
    # 1. 各種設定
    BATCH_SIZE = 64
    MODEL_PATH = "vit_mnist_model.pth"
    SAVE_DIR = "mistakes"  # 間違えた画像の保存先
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 2. モデルの準備
    model = ViT(
        image_size=28, patch_size=7, n_classes=10, channels=1,
        dim=128, depth=6, n_heads=8, mlp_dim=256
    )
    # 重みをロード
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Loaded: {MODEL_PATH}")
    else:
        print("Error: Model file not found.")
        return

    model.eval()

    # 3. データの準備
    data_manager = MNISTDataLoader(batch_size=BATCH_SIZE)
    test_loader = data_manager.get_test()

    # 4. 評価 & 誤分類の保存
    correct = 0
    total = 0
    mistake_count = 0
    max_save_mistakes = 10 # 保存する最大枚数（多すぎると大変なので）

    print("Evaluating...")
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # 正解数のカウント
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 誤分類箇所の特定
            mask = (predicted != labels)
            if mask.any():
                # 間違えた画像、予測値、正解ラベルを抽出
                wrong_images = images[mask]
                wrong_preds = predicted[mask]
                wrong_labels = labels[mask]

                for i in range(len(wrong_images)):
                    if mistake_count >= max_save_mistakes:
                        break
                    
                    # 保存処理
                    img_show = wrong_images[i][0] * 0.5 + 0.5
                    pred_v = wrong_preds[i].item()
                    true_v = wrong_labels[i].item()

                    plt.figure(figsize=(3, 3))
                    plt.imshow(img_show, cmap='gray')
                    plt.title(f"Pred: {pred_v} / True: {true_v}")
                    plt.axis('off')

                    save_path = f"{SAVE_DIR}/mistake_{mistake_count}_P{pred_v}_T{true_v}.png"
                    plt.savefig(save_path)
                    plt.close()
                    
                    print(f"Saved mistake: {save_path}")
                    mistake_count += 1

    accuracy = 100 * correct / total
    print("-" * 30)
    print(f"Final Accuracy: {accuracy:.2f}%")
    print(f"Total mistakes saved: {mistake_count}")

if __name__ == "__main__":
    run_inference()