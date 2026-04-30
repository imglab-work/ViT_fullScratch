import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from ViT import ViT
from models.MNISTDataLoader import MNISTDataLoader

import importlib.util
import sys
import time
import shutil

class ViTEvaluator:
    def __init__(self, model_path, conf_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. conf_path から Config クラスを動的にロードする
        spec = importlib.util.spec_from_file_location("config_module", conf_path)
        config_module = importlib.util.module_from_spec(spec)
        sys.modules["config_module"] = config_module
        spec.loader.exec_module(config_module)
        
        # ロードしたモジュールから Config クラスを取得
        loaded_config = config_module.Config

        # 2. ロードした設定（loaded_config）を使ってモデルをインスタンス化
        self.model = ViT(
            image_size=loaded_config.IMAGE_SIZE,
            patch_size=loaded_config.PATCH_SIZE,
            n_classes=loaded_config.N_CLASSES,
            channels=loaded_config.CHANNELS,
            dim=loaded_config.DIM,
            depth=loaded_config.DEPTH,
            n_heads=loaded_config.N_HEADS,
            mlp_dim=loaded_config.MLP_DIM
        ).to(self.device)
        
        # 重みのロード
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded weights from: {model_path}")
            print(f"✅ Loaded config from: {conf_path}")
        else:
            raise FileNotFoundError(f"Error: {model_path} not found.")
        
        self.model.eval()

    def evaluate(self, test_loader, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        mistakes_dir = os.path.join(save_dir, "mistakes")
        os.makedirs(mistakes_dir, exist_ok=True)

        all_preds, all_labels = [], []
        mistake_count = 0
        max_save_mistakes = 10
        attention_saved = False # 最初の1枚だけ保存するためのフラグ

        print("🚀 Evaluating...")
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs, attentions = self.model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # --- 1. 最初の1枚だけAttentionを記録 ---
                if not attention_saved:
                    # ファイル名を指定して保存
                    attn_path = os.path.join(save_dir, "attention_sample.png")
                    self.visualize_attention(images[0:1], attentions, attn_path)
                    attention_saved = True

                # --- 2. 誤分類画像の保存 & その時のAttentionも保存 ---
                mask = (predicted != labels)
                if mask.any() and mistake_count < max_save_mistakes:
                    # 誤分類したサンプルのインデックスを取得
                    indices = torch.where(mask)[0]
                    for idx in indices:
                        if mistake_count >= max_save_mistakes: break
                        
                        # 画像と予測・正解を抽出
                        img, p, t = images[idx:idx+1], predicted[idx].item(), labels[idx].item()
                        
                        # 誤分類の証拠写真
                        img_path = os.path.join(mistakes_dir, f"mistake_{mistake_count}_P{p}_T{t}.png")
                        self._save_mistake_img(img[0], p, t, img_path)
                        
                        # 【重要】その時のAttention Mapも並べて保存！
                        attn_mistake_path = os.path.join(mistakes_dir, f"mistake_{mistake_count}_attn.png")
                        self.visualize_attention(img, attentions, attn_mistake_path, idx=int(idx.item()))
                        
                        mistake_count += 1

        self._report_metrics(all_labels, all_preds, save_dir)


    def visualize_attention(self, img, attentions, save_path, idx=0):
        """
        idx: バッチ内の何番目の画像を表示するか
        """
        # 指定したインデックスのアテンションを取り出す
        # attentions[-1].shape -> [Batch, Head, N+1, N+1]
        last_attn = attentions[-1][idx] 
        
        mean_attn = last_attn.mean(dim=0) 
        cls_attn = mean_attn[0, 1:] 
        
        n = int(np.sqrt(cls_attn.size(0)))
        attn_map = cls_attn.reshape(n, n).cpu().numpy()

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        # 左：元画像 (正規化を戻して表示)
        orig_img = img[0, 0].cpu().numpy() * 0.5 + 0.5
        ax[0].imshow(orig_img, cmap='gray')
        ax[0].set_title("Original MNIST")
        ax[0].axis('off')

        # 右：アテンションを重ねる
        ax[1].imshow(orig_img, cmap='gray')
        # interpolation='bilinear' でパッチの境界を滑らかにする
        im = ax[1].imshow(attn_map, cmap='jet', alpha=0.5, extent=(0, 28, 28, 0), interpolation='bilinear')
        ax[1].set_title("Attention Heatmap")
        ax[1].axis('off')

        plt.savefig(save_path)
        plt.close()


    def _save_mistake_img(self, img_tensor, pred, true, save_path):
            # メソッドの引数をパス指定に変更して汎用化
            img_show = img_tensor[0].cpu().numpy() * 0.5 + 0.5
            plt.figure(figsize=(3, 3))
            plt.imshow(img_show, cmap='gray')
            plt.title(f"Pred: {pred} / True: {true}")
            plt.axis('off')
            plt.savefig(save_path)
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
    print("--- ViT Evaluation Mode ---")
    model_path = input("Enter weight file path: ").strip()
    config_path = input("Enter config file path: ").strip()
    
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print("❌ Error: File not found.")
        return

    # 1. 評価用の新しいディレクトリを作成
    timestamp = time.strftime("%Y%m%d_%H%M_eval")
    new_save_dir = f"results/{timestamp}"
    os.makedirs(new_save_dir, exist_ok=True)
    print(f"📂 New evaluation results will be saved in: {new_save_dir}")

    # 2. 入力された重みとConfigを新しいフォルダにコピーして保存
    # ファイル名が変わらないように os.path.basename を使います
    shutil.copy(model_path, os.path.join(new_save_dir, os.path.basename(model_path)))
    # configは常に「config_backup.py」という名前に統一して保存すると後で扱いやすいです
    new_conf_path = os.path.join(new_save_dir, "config_backup.py")
    shutil.copy(config_path, new_conf_path)
    print(f"📄 Copied weights and config to the new folder.")

    # 3. 評価クラスのインスタンス化（コピーした方のConfigを使う）
    try:
        evaluator = ViTEvaluator(model_path=model_path, conf_path=new_conf_path)
    except Exception as e:
        print(f"❌ Failed to initialize evaluator: {e}")
        return

    # 4. データのロード
    data_manager = MNISTDataLoader(batch_size=64) 
    test_loader = data_manager.get_test()

    # 5. 評価実行
    evaluator.evaluate(test_loader, save_dir=new_save_dir)
    
    print(f"✨ Evaluation complete! Check the results in {new_save_dir}")

if __name__ == "__main__":
    run_inference()