from torchvision.models import vit_b_16
import torch
import torch.nn as nn
import math

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size, seq_len, embed_dim = query.shape
        assert embed_dim == self.embed_dim
        assert key.shape == value.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)

        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)

        return attn_output

class MLPBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)

        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.ln_1 = nn.LayerNorm(embed_dim)
        self.self_attention = MultiheadAttention(embed_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, mlp_dim, dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = self.ln_1(x)
        x = self.self_attention(x, x, x, attn_mask, key_padding_mask)
        x = x + x

        x = self.ln_2(x)
        x = self.mlp(x)
        x = x + x

        return x
    

class Encoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, mlp_dim, dropout=0.):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.dropout = nn.Dropout(dropout)
        self.layers = nn.Sequential(*[
            EncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        for encoder in self.layers:
            x = encoder(x, attn_mask, key_padding_mask)

        x = self.ln(x)

        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size, image_channel, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.image_size = image_size
        self.image_channel = image_channel
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        assert image_size % patch_size == 0, 'image size must be divisible by patch size'
        num_patches = (image_size // patch_size) ** 2

        self.patch_size = patch_size
        self.to_patch_embedding = nn.Conv2d(
            self.image_channel, self.dim, self.patch_size, self.patch_size
        )

        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.encoders = Encoder(depth, dim, heads, mlp_dim, dropout)

        self.ln = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        assert height == width == self.image_size, 'image must be square'
        assert channels == 3, 'images must have 3 channels'

        x = self.to_patch_embedding(x)
        x = x.view(batch_size, -1, self.dim)
        x = torch.cat((self.cls_token.expand(batch_size, -1, -1), x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.encoders(x)
        x = self.ln(x[:, 0])
        x = self.fc(x)

        return x
    

model = VisionTransformer(
    image_size=32,
    image_channel=3,
    patch_size=4,
    num_classes=1000,
    dim=768,
    depth=12,
    heads=12,
    mlp_dim=3072,
    dropout=0.1,
)

rand_image = torch.rand(1, 3, 32, 32)

output = model(rand_image)