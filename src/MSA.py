import torch
import torch.nn as nn
import math
class MSA(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads # 1つのヘッドが持つ次元
        
        # Q, K, Vをまとめて作るための層
        self.qkv = nn.Linear(dim, dim * 3)
        # 最後に全てのヘッドを合体させた後に通す層
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        
        # 1. Q, K, Vを一気に生成 [B, N, D*3]
        qkv = self.qkv(x)
        
        # 2. Q, K, Vに切り分け、さらにマルチヘッド用に分割
        # [B, N, D*3] -> [B, N, 3, n_heads, head_dim]
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_dim)
        # 3. 軸を入れ替えて [3, B, n_heads, N, head_dim] にする
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # 4. それぞれを独立した変数に取り出す
        q, k, v = qkv[0], qkv[1], qkv[2] # それぞれ [B, n_heads, N, head_dim]

        # 5. スコア計算 (QK^T)
        # k.transpose(-2, -1) で [B, n_heads, head_dim, N] にして行列掛け算
        # 点積（内積）を計算 [B, n_heads, N, N]
        dots = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 6. Softmaxで「重み」にする
        attn = dots.softmax(dim=-1) # [B, n_heads, N, N]
        
        # 7. 重みに基づいてValueを足し合わせる
        # [B, n_heads, N, N] @ [B, n_heads, N, head_dim] -> [B, n_heads, N, head_dim]
        out = attn @ v
        
        # 8. 全てのヘッドを結合して元に戻す
        # [B, n_heads, N, head_dim] -> [B, N, n_heads, head_dim] -> [B, N, D]
        out = out.transpose(1, 2).reshape(B, N, D)
        
        return self.out(out)