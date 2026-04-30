import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from ViT import ViT
from models.MNISTDataLoader import MNISTDataLoader

class ViTEvaluator:
    def __init__(self, model_path, image_size=28, patch_size=7, dim=128, depth=6, n_heads=8, mlp_dim=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViT(
            image_size=image_size, patch_size=patch_size, n_classes=10, channels=1,
            dim=dim, depth=depth, n_heads=n_heads, mlp_dim=mlp_dim
        ).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded: {model_path}")
        else:
            raise FileNotFoundError(f"Error: {model_path} not found.")
        
        self.model.eval()

    def evaluate(self, test_loader, save_dir="analysis"):
        os.makedirs(save_dir, exist_ok=True)
        mistakes_dir = os.path.join(save_dir, "mistakes")
        os.makedirs(mistakes_dir, exist_ok=True)

        all_preds = []
        all_labels = []
        mistake_count = 0
        max_save_mistakes = 10

        print("🚀 Evaluating...")
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # --- 誤分類画像の保存 (既存ロジック) ---
                mask = (predicted != labels)
                if mask.any() and mistake_count < max_save_mistakes:
                    for img, p, t in zip(images[mask], predicted[mask], labels[mask]):
                        if mistake_count >= max_save_mistakes: break
                        self._save_mistake_img(img, p.item(), t.item(), mistake_count, mistakes_dir)
                        mistake_count += 1

        # 各種指標の計算
        self._report_metrics(all_labels, all_preds, save_dir)

    def _save_mistake_img(self, img_tensor, pred, true, count, save_dir):
        img_show = img_tensor[0].cpu().numpy() * 0.5 + 0.5
        plt.figure(figsize=(3, 3))
        plt.imshow(img_show, cmap='gray')
        plt.title(f"Pred: {pred} / True: {true}")
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"mistake_{count}_P{pred}_T{true}.png"))
        plt.close()

    def _report_metrics(self, labels, preds, save_dir):
        labels, preds = np.array(labels), np.array(preds)
        
        # 1. 正答率
        accuracy = (labels == preds).mean() * 100
        print("-" * 30)
        print(f"📊 Final Accuracy: {accuracy:.2f}%")

        # 2. 混合行列の作成
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=range(10), yticklabels=range(10))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix (Acc: {accuracy:.2f}%)")
        
        save_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()
        print(f"🎨 Saved Confusion Matrix: {save_path}")

def run_inference():
    # 各種設定
    MODEL_PATH = "vit_mnist_model.pth"
    
    # 評価クラスのインスタンス化
    evaluator = ViTEvaluator(model_path=MODEL_PATH)
    
    # データのロード
    data_manager = MNISTDataLoader(batch_size=64)
    test_loader = data_manager.get_test()
    
    # 評価実行
    evaluator.evaluate(test_loader)

if __name__ == "__main__":
    run_inference()