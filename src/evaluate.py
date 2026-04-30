import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from ViT import ViT
import time
from typing import Any, cast
from utils.config_loader import load_config_class

class ViTEvaluator:
    def __init__(self, model_path, conf_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. conf_path から Config クラスを動的にロードする
        Config = load_config_class(conf_path)
        # 2. ロードした設定（loaded_config）を使ってモデルをインスタンス化
        self.model = ViT(
            image_size=Config.IMAGE_SIZE,
            patch_size=Config.PATCH_SIZE,
            n_classes=Config.N_CLASSES,
            channels=Config.CHANNELS,
            dim=Config.DIM,
            depth=Config.DEPTH,
            n_heads=Config.N_HEADS,
            mlp_dim=Config.MLP_DIM
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
        # 成果物を入れるための「output」フォルダを作成
        output_dir = os.path.join(save_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # 間違い画像フォルダも output の中に作成
        mistakes_dir = os.path.join(output_dir, "mistakes")
        #os.makedirs(mistakes_dir, exist_ok=True)

        all_preds, all_labels = [], []
        all_confidences = []
        mistake_count = 0
        max_save_mistakes = 10
        attention_saved = False

        print("🚀 Evaluating...")
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs, attentions = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                conf, predicted = torch.max(probs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(conf.cpu().numpy())

                # --- 1. 最初の1枚のAttentionを記録 (outputフォルダ直下) ---
                if not attention_saved:
                    attn_path = os.path.join(output_dir, "attention_sample.png")
                    self.visualize_attention(images[0:1], attentions, attn_path)
                    attention_saved = True

                # --- 2. 誤分類画像 & Attention (output/mistakesフォルダ内) ---
                mask = (predicted != labels)
                if mask.any() and mistake_count < max_save_mistakes:
                    indices = torch.where(mask)[0]
                    for idx in indices:
                        if mistake_count >= max_save_mistakes: break
                        
                        img, p, t = images[idx:idx+1], predicted[idx].item(), labels[idx].item()
                        
                        img_path = os.path.join(mistakes_dir, f"mistake_{mistake_count}_P{p}_T{t}.png")
                        #self._save_mistake_img(img[0], p, t, img_path)
                        
                        attn_mistake_path = os.path.join(mistakes_dir, f"mistake_{mistake_count}_attn.png")
                        #self.visualize_attention(img, attentions, attn_mistake_path, idx=int(idx.item()))
                        
                        mistake_count += 1

        # 混合行列などのレポートも output フォルダ内に出力
        self._report_metrics(all_labels, all_preds, all_confidences, output_dir)


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

    def _report_metrics(self, labels, preds, confs, output_dir):
        labels, preds, confs = np.array(labels), np.array(preds), np.array(confs)
        accuracy = (labels == preds).mean() * 100
        
        # 1. 確信度の統計計算 (0.0〜1.0 の範囲を想定)
        correct_mask = (labels == preds)
        mistake_mask = ~correct_mask
        
        avg_conf_total = confs.mean()
        avg_conf_correct = confs[correct_mask].mean() if any(correct_mask) else 0
        avg_conf_mistake = confs[mistake_mask].mean() if any(mistake_mask) else 0

        # 2. classification_reportを辞書形式で取得してMarkdown化
        report_dict = classification_report(labels, preds, output_dict=True)
        report_dict = cast(dict[str, Any], classification_report(labels, preds, output_dict=True))

        report_path = os.path.join(output_dir, "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Overall Accuracy**: `{accuracy:.2f}%` \n\n")
            
            f.write(f"## 📊 Confidence Analysis\n\n")
            f.write(f"| Category | Average Confidence |\n")
            f.write(f"| :--- | :--- |\n")
            f.write(f"| **Total** | {avg_conf_total:.4f} |\n")
            f.write(f"| ✅ **Correct** | {avg_conf_correct:.4f} |\n")
            f.write(f"| ❌ **Mistake** | {avg_conf_mistake:.4f} |\n\n")
            
            f.write(f"## 📈 Metrics per Class\n\n")
            f.write(f"| Class | Precision | Recall | F1-score | Avg Conf |\n")
            f.write(f"| :--- | :--- | :--- | :--- | :--- |\n")
            for i in range(10):
                label = str(i)
                m = report_dict[label]
                # そのクラスだけの平均確信度を計算
                c_conf = confs[labels == i].mean() if any(labels == i) else 0
                f.write(f"| Digit {i} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {c_conf:.4f} |\n")
            
            f.write(f"\n## 🏁 Summary\n\n")
            f.write(f"| Type | Precision | Recall | F1-score |\n")
            f.write(f"| :--- | :--- | :--- | :--- |\n")
            f.write(f"| Macro Avg | {report_dict['macro avg']['precision']:.4f} | {report_dict['macro avg']['recall']:.4f} | {report_dict['macro avg']['f1-score']:.4f} |\n")
            f.write(f"| Weighted Avg | {report_dict['weighted avg']['precision']:.4f} | {report_dict['weighted avg']['recall']:.4f} | {report_dict['weighted avg']['f1-score']:.4f} |\n")
        
        print(f"📝 Markdown report saved: {report_path}")

        # 4. 混合行列の作成 (既存の処理)
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=range(10), yticklabels=range(10))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix (Acc: {accuracy:.2f}%)")
        
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()

