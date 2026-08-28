import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Embedding senoidal para o timestep de difusão/flow matching t.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: (B,) ou (B, 1) com valores entre 0 e 1
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x shape: (B, num_heads, T, dim_head)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:x.shape[2], :]
    while freqs_cis.ndim < x_complex.ndim:
        freqs_cis = freqs_cis.unsqueeze(0)
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_out.type_as(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, D)
        B, T, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # Scaled dot-product attention
        if hasattr(F, "scaled_dot_product_attention"):
            attn_mask = None
            if mask is not None:
                # mask: (B, T) -> (B, 1, 1, T)
                attn_mask = mask.unsqueeze(1).unsqueeze(2)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            if mask is not None:
                mask_val = torch.finfo(dots.dtype).min
                dots = dots.masked_fill(~mask.unsqueeze(1).unsqueeze(2), mask_val)
            attn = F.softmax(dots, dim=-1)
            out = torch.matmul(attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    """
    Bloco DiT com modulação AdaLN (Adaptive Layer Norm) condicionada pelo timestep t.
    """
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)

        # 6 scale-shift modulations: gamma1, beta1, alpha1, gamma2, beta2, alpha2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # t_emb: (B, D)
        mod = self.adaLN_modulation(t_emb).unsqueeze(1)  # (B, 1, 6*D)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        # Attention block with modulation
        norm_x1 = self.norm1(x) * (1 + gamma1) + beta1
        x = x + alpha1 * self.attn(norm_x1, mask=mask)

        # FeedForward block with modulation
        norm_x2 = self.norm2(x) * (1 + gamma2) + beta2
        x = x + alpha2 * self.ff(norm_x2)
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) completo para Flow Matching acústico.
    Prediz o vetor de velocidade vetorial v_t(x_t, t) para guiar o fluxo mel.
    """
    def __init__(
        self,
        dim: int = 512,
        depth: int = 8,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        mel_dim: int = 100,
        text_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.mel_dim = mel_dim

        # Projeção de entrada: recebe mel ruidoso x_t concatenado com mel de prompt e texto
        # Entrada: [x_t (mel_dim) + prompt_mel (mel_dim) + text_features (text_dim)]
        in_dim = mel_dim * 2 + text_dim
        self.input_proj = nn.Linear(in_dim, dim)

        # Embedding temporal
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        # Camadas DiT Transformer
        self.blocks = nn.ModuleList([
            DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout)
            for _ in range(depth)
        ])

        # Projeção final para velocidade do fluxo Mel (mel_dim)
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True)
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.final_proj = nn.Linear(dim, mel_dim)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prompt_mel: torch.Tensor,
        text_features: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x_t: (B, T, mel_dim) - mel no instante t do flow
        t: (B,) ou (B, 1) - tempo do flow matching [0, 1]
        prompt_mel: (B, T, mel_dim) - áudio de referência / prompt
        text_features: (B, T, text_dim) - representação alinhada do texto
        mask: (B, T) booleano
        """
        # Concatena inputs na dimensão dos canais de feature
        inputs = torch.cat([x_t, prompt_mel, text_features], dim=-1)
        h = self.input_proj(inputs)  # (B, T, dim)

        # Calcula embedding temporal
        t_emb = self.time_mlp(t.squeeze(-1) if t.ndim > 1 else t)  # (B, dim)

        # Passa pelos blocos DiT
        for block in self.blocks:
            h = block(h, t_emb, mask=mask)

        # Projeção final com AdaLN
        scale, shift = self.final_adaLN(t_emb).unsqueeze(1).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale) + shift
        v_pred = self.final_proj(h)  # (B, T, mel_dim)
        
        if mask is not None:
            v_pred = v_pred * mask.unsqueeze(-1).to(v_pred.dtype)
        return v_pred
