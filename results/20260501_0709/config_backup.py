class Config:
    # --- 基本設定 ---
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 1e-4

    # --- モデル（ViT）の構造設定 ---
    #IMAGE_SIZEはPATCH_SIZEで割り切れる必要がある
    IMAGE_SIZE = 28
    PATCH_SIZE = 14
    CHANNELS = 1
    N_CLASSES = 10
    
    DIM = 128
    DEPTH = 6
    N_HEADS = 8
    MLP_DIM = 256
