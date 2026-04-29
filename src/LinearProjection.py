import torch
import torch.nn as nn
import math
import numpy as np

#抽象的なベクトルに変換する
class LinearProjection(nn.Module):
    def __init__(self, patch_dim, dim):
        super().__init__()
        # 1. 重み (Weight) と バイアス (Bias) を自分で定義
        # 行列のサイズは (入力次元, 出力次元)
        self.W = nn.Parameter(torch.randn(patch_dim, dim)) 
        self.b = nn.Parameter(torch.zeros(dim))
        
        # 重みの初期化（これを行わないと学習が安定しません）
        # nn.Linearと同じように、入力サイズの逆数でスケーリングするのが一般的
        std = 1.0 / math.sqrt(patch_dim)
        nn.init.uniform_(self.W, -std, std)
        nn.init.uniform_(self.b, -std, std)
        
    #(B,N,patch_dim)×(patch_dim,dim)=(B,N,dim)
    def forward(self, x):
        # x.shape: [B, N, D]
        x = x @ self.W + self.b
        # x.shape: [batch_size, n_patches, dim]
        return x