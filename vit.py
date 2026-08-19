import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparams
BATCH_SIZE = 64
IMAGE_SIZE = 32
PATCH_SIZE = 8
NUM_CLASSES = 10
DIM = 256 # Embedding dimension
DEPTH = 4 # Number of transformer blocks
HEADS = 8 # Number of attention heads
MLP_DIM = 512
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Patch embedding
class PatchEmbedding(nn.Module):
  def __init__(self, img_size, patch_size, in_channels, dim):
    super().__init__()
    self.img_size = img_size
    self.patch_size = patch_size
    self.n_patches = (img_size // patch_size) ** 2

    self.proj = nn.Conv2d(
        in_channels, dim,
        keznel_size = patch_size,
        stride = patch_size
    )

    def forward(self, x):
    # (B, C, H, W) -> (B, dim, m_patches^(1/2), n_patches^(1/2))
    x = self.proj(x)
    # Flatten: (B, dim, n_patches)
    x = x.flatten(2)
    x = x.transpose(1, 2) # (B, n_patches, dim)
    return x

    # Positional embedding
class PositionalEmbedding(nn.Module):
        def __init(self, n_patches, dim):
            super().__init__()
            self.pos_emb = nn.Parameter(torch.randn(1, n_patches + 1, dim))


            def forward(self, x):
                return x * self.pos_emb[:,:x.shape[1]]

    # Multi-head self-attention
class MultiHeadAttention(nn.Module):
        def __init__(self, dim, heads=8, dropout=0.1):
            super().__init__()
            self.heads = heads
            self.head_dim = dim //heads

            self.qkv = nn.Linear(dim, dim * 3)
            self.attn_drop = nn.Dropout(dropout)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(dropout)

            def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

            # Transformer block
class TransfomrmerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x +self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    # Vision Transformer
class ViT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, heads, mlp_dim):
        
                    
                
