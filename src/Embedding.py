import torch
import torch.nn as nn


#クラストークンの追加と位置エンコーディング
class Embedding(nn.Module):
    def __init__(self, dim, n_patches):
        super().__init__()
        # 1. [class] トークン (学習可能なベクトル)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # 2. 位置エンコーディング (学習可能なベクトル)
        # パッチ数 + クラストークン分 (n_patches + 1) 必要
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, dim))
    #【入力】[B,N,dim]のテンソル(画像の特徴を含んでいる)
    #【出力】[B,N+1,dim]のテンソル(先頭にクラストークンを追加+位置情報を加算)
    def forward(self, x):
        # x.shape: [B, N, dim]
        B, N, D = x.shape

        # クラストークンをバッチサイズ分コピーして、先頭に結合
        # (1, 1, dim) -> (B, 1, dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # (B, N, dim) -> (B, N+1, dim)
        x = torch.cat((cls_tokens, x), dim=1)

        # 位置エンコーディングを足す
        # (B, N+1, dim) + (1, N+1, dim) -> Broadcastingで各バッチに足される
        x = x + self.pos_embedding

        return x