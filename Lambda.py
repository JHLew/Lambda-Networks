import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

'''
Lambda Networks: Modeling Long-Range Interactions without Attention
https://arxiv.org/abs/2102.08602

Lambda Layers & Lambda Convolution.
referenced 
https://github.com/lucidrains/lambda-networks
https://github.com/d-li14/lambda.pytorch
at some points.
'''


class LambdaLayer(nn.Module):
    def __init__(self, dim, dim_k=16, dim_u=1, heads=4, norm=nn.BatchNorm2d, ff_sigma=10):
        super(LambdaLayer, self).__init__()
        self.dim = dim
        self.dim_k = dim_k
        self.dim_u = dim_u
        self.heads = heads
        assert dim % heads == 0, 'number of dimensions should be divisible by number of heads'
        self.dim_v = dim // heads

        self.conv_qkv = nn.Conv2d(dim, dim_k * heads + dim_k * dim_u + self.dim_v * dim_u, 1, bias=False)
        self.norm_q = norm(dim_k * heads)
        self.norm_v = norm(self.dim_v * dim_u)
        self.norm_k = nn.Softmax(dim=-1)

        spatial_dim = 2
        assert dim_k % 2 == 0, 'number of dimension k must be divisible by 2'
        self.ff = nn.Parameter(torch.randn(spatial_dim, dim_k // 2) * ff_sigma)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def pos_embedding(self, size, _range=(-1, 1)):
        h, w = size
        _from, _to = _range
        coords = [torch.linspace(_from, _to, steps=h), torch.linspace(_from, _to, steps=w)]
        mgrid = torch.stack(torch.meshgrid(*coords), dim=-1).to(self.device)
        rel_pos = rearrange(mgrid, 'h w c -> (h w) c')
        pos_proj = torch.einsum('m c, c k -> m k', rel_pos, self.ff) * 2 * np.pi
        ff_rel_pos = torch.cat([torch.sin(pos_proj), torch.cos(pos_proj)], dim=1)

        return ff_rel_pos

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.conv_qkv(x)
        q, k, v = torch.split(qkv, [self.dim_k * self.heads, self.dim_k * self.dim_u, self.dim_v * self.dim_u], dim=1)
        q = rearrange(self.norm_q(q), 'b (k n) h w -> b n k (h w)', k=self.dim_k, n=self.heads)
        v = rearrange(self.norm_v(v), 'b (v u) h w -> b u v (h w)', v=self.dim_v, u=self.dim_u)
        k = self.norm_k(rearrange(k, 'b (u k) h w -> b u k (h w)', u=self.dim_u, k=self.dim_k))
        e = self.pos_embedding(size=(h, w))

        content_lambda = torch.einsum('b u k m, b u v m -> b k v', k, v)
        position_lambda = torch.einsum('m k, b u v m -> b k v', e, v)

        content_output = torch.einsum('b n k m, b k v -> b n v m', q, content_lambda)
        position_output = torch.einsum('b n k m, b k v -> b n v m', q, position_lambda)
        output = rearrange(content_output + position_output, 'b n v (h w) -> b (n v) h w', h=h, w=w)

        return output


class LambdaConv(nn.Module):
    def __init__(self, dim, dim_k=16, dim_u=1, heads=4, scope=23, norm=nn.BatchNorm2d):
        super(LambdaConv, self).__init__()
        self.dim = dim
        self.dim_k = dim_k
        self.dim_u = dim_u
        self.heads = heads
        assert dim % heads == 0, 'number of dimensions should be divisible by number of heads'
        self.dim_v = dim // heads

        self.conv_qkv = nn.Conv2d(dim, heads * dim_k + dim_k * dim_u + self.dim_v * dim_u, 1, bias=False)
        self.norm_q = norm(heads * dim_k)
        self.norm_v = norm(self.dim_v * dim_u)
        self.norm_k = nn.Softmax(dim=-1)
        self.lambda_conv = nn.Conv3d(dim_u, dim_k, (1, scope, scope), padding=(0, (scope - 1) // 2, (scope - 1) // 2))

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.conv_qkv(x)
        q, k, v = torch.split(qkv, [self.heads * self.dim_k, self.dim_k * self.dim_u, self.dim_v * self.dim_u], dim=1)
        q = rearrange(self.norm_q(q), 'b (k n) h w -> b n k (h w)', k=self.dim_k, n=self.heads)
        v = rearrange(self.norm_v(v), 'b (v u) h w -> b u v (h w)', v=self.dim_v, u=self.dim_u)
        k = self.norm_k(rearrange(k, 'b (u k) h w -> b u k (h w)', u=self.dim_u, k=self.dim_k))

        content_lambda = torch.einsum('b u k m, b u v m -> b k v', k, v)
        position_lambda = rearrange(self.lambda_conv(rearrange(v, 'b u v (h w) -> b u v h w', h=h, w=w)),
                                    'b u v h w -> b u v (h w)')
        content_output = torch.einsum('b n k m, b k v -> b n v m', q, content_lambda)
        position_output = torch.einsum('b n k m, b k v m -> b n v m', q, position_lambda)
        output = rearrange(content_output + position_output, 'b n v (h w) -> b (n v) h w', h=h, w=w)

        return output
