import torch
import torch.nn as nn
import math

class MSA(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        # 特徴量の数dim(多い)を、n_deads個のグループに分けて処理する。
        # 1グループあたりが担う特徴量の数がhead_dim。
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads

        # 1. Q, K, V を一気に生成するための重み
        # [dim, dim * 3] という一つの巨大な行列にする
        # パッチの特徴量(dim)を、3人分(QKV)の全ヘッド合計次元(dim*3)へ飛ばす
        self.w_qkv = nn.Parameter(torch.randn(dim, dim * 3))
        self.b_qkv = nn.Parameter(torch.zeros(dim * 3))

        # 2. 最後に全てのヘッドを合体させた後に通す重み
        # 全ヘッドが持ち寄った情報(dim)を、最終的な特徴量(dim)へ再構築する
        self.w_out = nn.Parameter(torch.randn(dim, dim))
        self.b_out = nn.Parameter(torch.zeros(dim))

        # 初期化
        std = 1.0 / math.sqrt(dim)
        nn.init.uniform_(self.w_qkv, -std, std)
        nn.init.uniform_(self.w_out, -std, std)

    def forward(self, x):
        B, N, D = x.shape

        # --- 手順1: Q, K, V の生成 ---
        # 行列演算: [B, N, D] @ [D, D*3] + [D*3] = [B, N, D*3]
        qkv = x @ self.w_qkv + self.b_qkv

        # --- 手順2: 形状の変形 (マルチヘッド化) ---
        # [B, N, D*3] -> [B, N, 3, n_heads, head_dim]
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_dim)
        # 軸の入れ替え [3, B, n_heads, N, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # --- 手順3: スコア計算 (Attention Score) ---
        # Q[B, h, N, d_h] @ K_T[B, h, d_h, N] = dots[B, h, N, N]
        dots = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = dots.softmax(dim=-1)

        # --- 手順4: 重み付き合計 (Value の集約) ---
        # attn[B, h, N, N] @ V[B, h, N, d_h] = out[B, h, N, d_h]
        out = attn @ v

        # --- 手順5: ヘッドの結合と出力変換 ---
        # 軸を戻して結合 [B, N, n_heads, head_dim] -> [B, N, D]
        out = out.transpose(1, 2).reshape(B, N, D)
        
        # 最後の線形変換: [B, N, D] @ [D, D] + [D] = [B, N, D]
        out = out @ self.w_out + self.b_out
        
        return out, attn