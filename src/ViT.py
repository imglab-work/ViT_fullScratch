import torch
import torch.nn as nn
from models.Patching import Patching
from models.LinearProjection import LinearProjection
from models.Embedding import Embedding
from models.TransformerEncoder import TransformerEncoder
from models.MLPHead import MLPHead

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, n_classes, dim, depth, n_heads, channels = 3, mlp_dim = 256):
        """ [input]
            - image_size (int) : 画像の縦の長さ（= 横の長さ）
            - patch_size (int) : パッチの縦の長さ（= 横の長さ）
            - n_classes (int) : 分類するクラスの数
            - dim (int) : 各パッチのベクトルが変換されたベクトルの長さ（参考[1] (1)式 D）
            - depth (int) : Transformer Encoder の層の深さ（参考[1] (2)式 L）
            - n_heads (int) : Multi-Head Attention の head の数
            - chahnnels (int) : 入力のチャネル数（RGBの画像なら3）
            - mlp_dim (int) : MLP の隠れ層のノード数
        """

        super().__init__()
        
        # Params
        n_patches = (image_size // patch_size) ** 2 #パッチ総数（画像を何枚に切り分けたか）
        patch_dim = channels * patch_size * patch_size #パッチ1個あたりの生（なま）の情報量
        self.depth = depth

        # Layers
        self.patching = Patching(patch_size = patch_size)
        self.linear_projection_of_flattened_patches = LinearProjection(patch_dim = patch_dim, dim = dim)
        self.embedding = Embedding(dim = dim, n_patches = n_patches)

        self.transformer_encoder = TransformerEncoder(dim = dim, n_heads = n_heads, mlp_dim = mlp_dim, depth = depth)
        self.mlp_head = MLPHead(dim = dim, out_dim = n_classes)


    def forward(self, img):
        """ [input]
            - img (torch.Tensor) : 画像データ
                - img.shape = torch.Size([batch_size, channels, image_height, image_width])
        """

        x = img

        # 1. パッチに分割
        # x.shape : [batch_size, channels, image_height, image_width] -> [batch_size, n_patches, channels * (patch_size ** 2)]
        x = self.patching(x)

        # 2. 各パッチをベクトルに変換
        # x.shape : [batch_size, n_patches, channels * (patch_size ** 2)] -> [batch_size, n_patches, dim]
        x = self.linear_projection_of_flattened_patches(x)

        # 3. [class] トークン付加 + 位置エンコーディング 
        # x.shape : [batch_size, n_patches, dim] -> [batch_size, n_patches + 1, dim]
        x = self.embedding(x)

        # 4. Transformer Encoder
        # x.shape : No Change
        x = self.transformer_encoder(x)

        # 5. 出力の0番目のベクトルを MLP Head で処理
        # x.shape : [batch_size, n_patches + 1, dim] -> [batch_size, dim] -> [batch_size, n_classes]
        x = x[:, 0, :]#一応最後の「:」は省略可能
        x = self.mlp_head(x)

        return x