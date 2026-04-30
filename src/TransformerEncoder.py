import torch
import torch.nn as nn
from MSA import MSA
from MLP import MLP
class TransformerEncoder(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, depth):
        super().__init__()
        self.depth = depth
        # ブロックを何層も重ねるためにリスト化
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),                      # LN 1
                MSA(dim, n_heads),                      # MSA
                nn.LayerNorm(dim),                      # LN 2
                MLP(dim, mlp_dim)                       # MLP
            ]))

    #【入力】[B,N+1,dim]のテンソル(先頭にクラストークンを追加+位置情報を加算)
    #【出力】[B,N+1,dim]のテンソル(MSAにより関連性を見る・MLPにより活性化関数を通す)
    def forward(self, x):
        for ln1, msa, ln2, mlp in self.layers:
            # 1. MSA ステップ (残差接続)
            # x = x + self.msa(self.ln1(x)) と同じ意味
            x = x + msa(ln1(x))
            
            # 2. MLP ステップ (残差接続)
            # x = x + self.mlp(self.ln2(x)) と同じ意味
            x = x + mlp(ln2(x))
            
        return x