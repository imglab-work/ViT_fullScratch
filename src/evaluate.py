import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from ViT import ViT
import time

import importlib.util
import sys


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
                conf, predicted = torch.max(outputs, 1)

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
        
        # 1. 確信度の統計計算
        correct_mask = (labels == preds)
        mistake_mask = ~correct_mask
        
        avg_conf_total = confs.mean()
        avg_conf_correct = confs[correct_mask].mean() if any(correct_mask) else 0
        avg_conf_mistake = confs[mistake_mask].mean() if any(mistake_mask) else 0

        # 2. 数字ごとの確信度
        class_conf = {}
        for i in range(10):
            mask = (labels == i)
            class_conf[i] = confs[mask].mean() if any(mask) else 0

        # 3. レポートの作成と保存
        report = classification_report(
            labels, preds, 
            target_names=[f"Digit {i}" for i in range(10)],
            digits=4
        )

        report_path = os.path.join(output_dir, "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"--- ViT MNIST Evaluation Report ---\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Overall Accuracy: {accuracy:.2f}%\n\n")
            
            f.write(f"--- Confidence Analysis ---\n")
            f.write(f"Average Confidence (Total):   {avg_conf_total:.4f}\n")
            f.write(f"Average Confidence (Correct): {avg_conf_correct:.4f}\n")
            f.write(f"Average Confidence (Mistake): {avg_conf_mistake:.4f}\n\n")
            
            f.write(f"--- Confidence per Class ---\n")
            for i in range(10):
                f.write(f"Digit {i}: {class_conf[i]:.4f}\n")
            f.write("\n")
            
            f.write(f"--- Classification Report ---\n")
            f.write(str(report))
        
        print(f"📝 Saved Comprehensive Report: {report_path}")

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

