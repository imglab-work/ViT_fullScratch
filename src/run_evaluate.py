import time
import shutil
import glob
import os
import importlib.util
import sys
from models.MyDataLoader import MNISTDataLoader
from evaluate import ViTEvaluator
def run_inference():
    print("\n" + "="*30)
    print("🚀 ViT Evaluation Mode")
    print("="*30)
    
    # --- 1. フォルダパスの入力と自動検出 ---
    target_dir = input("Enter the experiment folder path (e.g., results/20260430_1628): ").strip()
    
    if not os.path.isdir(target_dir):
        print(f"❌ Error: Directory not found: {target_dir}")
        return

    # 重みファイル (*.pth) と Config (config_backup.py) を自動探索
    weight_files = glob.glob(os.path.join(target_dir, "*.pth"))
    config_files = glob.glob(os.path.join(target_dir, "config_backup.py"))

    if not weight_files:
        print("❌ Error: No .pth file found in the directory.")
        return
    if not config_files:
        print("❌ Error: config_backup.py not found in the directory.")
        return

    model_path = weight_files[0]  # 最初に見つかった重みを使用
    config_path = config_files[0]
    print(f"🔍 Found Weight: {os.path.basename(model_path)}")
    print(f"🔍 Found Config: {os.path.basename(config_path)}")

    #conf_path から Config クラスを動的にロードする
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config_module"] = config_module
    spec.loader.exec_module(config_module)
    loaded_config = config_module.Config

    # --- 2. 出力モードの選択 ---
    print("\nSelect Output Mode:")
    print("  [1] Create a NEW timestamped folder (Safe)")
    print("  [2] Overwrite / Add to the INPUT folder (Direct)")
    mode = input("Select mode (1 or 2): ").strip()

    if mode == "1":
        timestamp = time.strftime("%Y%m%d_%H%M_eval")
        save_dir = f"results/{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        
        # 新しいフォルダに重みとConfigをコピー
        shutil.copy(model_path, os.path.join(save_dir, os.path.basename(model_path)))
        new_conf_path = os.path.join(save_dir, "config_backup.py")
        shutil.copy(config_path, new_conf_path)
    else:
        # 入力されたフォルダをそのまま出力先にする
        save_dir = target_dir
        new_conf_path = config_path
        print(f"⚠️ Direct mode: Results will be saved in {target_dir}/output/")

    # --- 3. 評価実行 ---
    try:
        evaluator = ViTEvaluator(model_path=model_path, conf_path=new_conf_path)
        data_manager = MNISTDataLoader(batch_size=64, dataset=loaded_config.DATASET) 
        test_loader = data_manager.get_test()
        
        evaluator.evaluate(test_loader, save_dir=save_dir)
        
        print("\n" + "-"*30)
        print(f"✨ Success! Saved in: {save_dir}/output/")
        print("-"*30)
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    run_inference()