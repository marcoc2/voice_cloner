import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTokenizer:
    """
    Tokenizer baseado em caracteres UTF-8 com suporte completo a português,
    inglês, pontuação, acentuação e caracteres especiais.
    """
    def __init__(self):
        # 0: <PAD>, 1: <UNK>, 2: <BOS>, 3: <EOS>, 4: <MASK>
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.mask_token_id = 4
        
        # Caracteres padrão
        chars = (
            " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ"
            ".,!?;:'\"-—_()[]{}/\\@#$%^&*+=<>~`\n"
        )
        self.char_to_id = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
            "<MASK>": 4,
        }
        for idx, char in enumerate(chars, start=5):
            if char not in self.char_to_id:
                self.char_to_id[char] = len(self.char_to_id)
                
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = max(256, len(self.char_to_id))

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        for char in text:
            ids.append(self.char_to_id.get(char, self.unk_token_id))
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        res = []
        for i in ids:
            if i in (self.pad_token_id, self.bos_token_id, self.eos_token_id, self.mask_token_id):
                continue
            res.append(self.id_to_char.get(i, ""))
        return "".join(res)


class ConvNeXtBlock(nn.Module):
    """
    Bloco ConvNeXt 1D para extração de representações textuais robustas.
    """
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        ) if layer_scale_init_value > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, D)
        residual = x
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, T, D)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return residual + x


class TextEncoder(nn.Module):
    """
    Codificador de texto que converte tokens em sequências de representação
    acústico-semântica de alta dimensão.
    """
    def __init__(self, vocab_size: int = 256, embed_dim: int = 512, depth: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.blocks = nn.ModuleList([ConvNeXtBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.null_embedding = nn.Parameter(torch.randn(embed_dim))

    def forward(self, text_tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        text_tokens: (B, T_text)
        mask: (B, T_text) booleano (True para tokens válidos, False para pad)
        """
        x = self.embedding(text_tokens)  # (B, T_text, D)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x

    def align_to_mel_length(self, text_emb: torch.Tensor, target_mel_len: int) -> torch.Tensor:
        """
        Interpola ou expande a representação do texto para coincidir com a duração temporal do espectrograma mel.
        """
        # text_emb: (B, T_text, D) -> transpose para (B, D, T_text)
        x = text_emb.transpose(1, 2)
        x_interp = F.interpolate(x, size=target_mel_len, mode="linear", align_corners=False)
        return x_interp.transpose(1, 2)  # (B, target_mel_len, D)
