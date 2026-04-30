import torch.nn as nn
class MLPHead(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        # 1. データの正規化
        self.ln = nn.LayerNorm(dim)
        # 2. 最終的な分類器 (dim -> クラス数)
        self.fc = nn.Linear(dim, out_dim)

    def forward(self, x):
        # x.shape: [B, dim] (既に先頭トークンが取り出された状態)
        
        x = self.ln(x)
        x = self.fc(x)
        
        return x