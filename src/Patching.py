import torch
import torch.nn as nn
import numpy as np

#画像とミニ画像に分割して横に並べる
class Patching(nn.Module):
    def __init__(self, patch_size):
        """ [input]
            - patch_size (int) : パッチの縦の長さ（=横の長さ）
        """
        super().__init__()
        self.p = patch_size

    def forward(self,x):
        B, C, H, W = x.shape
        p = self.p #パッチサイズ
        # 1. 分割して軸を入れ替える
        x = x.reshape(B, C, H//p, p, W//p, p)#H//pの方向にp枚、W//pの方向にp枚ある#合計：1枚の画像がpxp枚のミニ画像に分割
        x = x.transpose(0, 2, 4, 3, 5, 1) # (B, H/p, W/p, p, p, C)
        # 2. 合体させてフラットにする
        x = x.reshape(B, (H//p) * (W//p), p*p*C)#(B,N,D)
        return x