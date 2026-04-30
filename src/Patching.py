import torch
import torch.nn as nn
import numpy as np

#画像をミニ画像に分割して横に並べる
class Patching(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.p = patch_size
    #【入力】[B,C,H,W]の複数枚画像のテンソル
    #【出力】[B,N,D]の、ミニ画像ベクトル(=パッチ)が並んだテンソル
    def forward(self, x):
        B, C, H, W = x.shape
        p = self.p
        
        # 1. 次元を細かく分解
        x = x.view(B, C, H // p, p, W // p, p)#H//pの方向にp枚、W//pの方向にp枚ある#合計：1枚の画像がpxp枚のミニ画像に分割
        # 2. 軸を並び替える (B, H/p, W/p, p, p, C)
        x = x.permute(0, 2, 4, 3, 5, 1)
        # 3. フラットに変形
        x = x.reshape(B, (H // p) * (W // p), p * p * C)
        return x