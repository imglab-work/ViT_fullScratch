import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        # 1. 1層目: dim -> mlp_dim
        self.fc1 = nn.Linear(dim, mlp_dim)
        # 2. 活性化関数: GELU (ReLUより少し滑らかな関数)
        self.gelu = nn.GELU()
        # 3. 2層目: mlp_dim -> dim (元のサイズに戻す)
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        # x.shape: [B, N+1, dim]
        
        x = self.fc1(x) # [B, N+1, mlp_dim] に広がる
        x = self.gelu(x)
        x = self.fc2(x) # [B, N+1, dim] に戻る
        
        return x