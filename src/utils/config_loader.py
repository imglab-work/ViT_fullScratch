import importlib.util
import sys
import os

def load_config_class(config_path):
    """
    指定されたパスのPythonファイルから Config クラスを動的にロードする。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")

    # モジュール名が衝突しないよう、ユニークな名前（または固定名）で読み込む
    module_name = "dynamic_config_module"
    
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    config_module = importlib.util.module_from_spec(spec)
    
    # sys.modules に登録することで、モジュール内での相対インポート等も安定する
    sys.modules[module_name] = config_module
    
    try:
        spec.loader.exec_module(config_module)
    except Exception as e:
        raise ImportError(f"❌ Failed to execute config module: {e}")

    # ロードされたモジュールから Config クラスを返す
    if hasattr(config_module, "Config"):
        return config_module.Config
    else:
        raise AttributeError(f"❌ 'Config' class not found in {config_path}")