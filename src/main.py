from ViT import ViT
import torch

# モデルのインスタンス化
model = ViT(
    image_size=224, 
    patch_size=16, 
    n_classes=10, 
    dim=128, 
    depth=6, 
    n_heads=8, 
    mlp_dim=256
)

# 偽の画像データ (Batch=1, RGB=3, Size=224x224)
img = torch.randn(1, 3, 224, 224)

# 予測
preds = model(img)
print(preds.shape) # torch.Size([1, 10]) と出れば成功です！