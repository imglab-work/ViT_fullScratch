import torch
import torch.nn as nn
import math
import numpy as np

from config import Config

#抽象的なベクトルに変換する
#全結合層であるが、CNNの全結合とは違い前半で使う。
#(CNNの全結合)畳み込み層で特徴を抽出しきった最後（出口）に、抽出した特徴から最終的な分類を行う
#(ViTの全結合)パッチという「生データ」を、Transformerが処理しやすい「概念的なベクトル」に変換する
#「dim」が特徴の数そのもの
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

        self.patch_shuffle = Config.PATCH_SHUFFLE
    
    #【入力】[B,N,D]のミニ画像ベクトル(=パッチ)が並んだテンソル
    #【出力】[B,N,dim]のテンソル(画像の特徴を含んでいる)
    #(B,N,patch_dim)×(patch_dim,dim)=(B,N,dim)
    #各パッチに対してそれぞれ同じ全結合を通す
    def forward(self, x):
        # x.shape: [B, N, D]
        x = x @ self.W + self.b
        # x.shape: [batch_size, n_patches, dim]
        
        #ベクトルをシャッフルするか？
        #結論：しない方がいい
        if self.patch_shuffle:
            b, n, _ = x.shape
            for i in range(b):
                indices = torch.randperm(n)
                x[i] = x[i, indices, :]
        return x