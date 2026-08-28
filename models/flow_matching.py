import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .dit import DiT
from .text_encoder import TextEncoder


class ConditionalFlowMatching(nn.Module):
    """
    Conditional Flow Matching (Optimal Transport CFM) para síntese e clonagem de voz.
    Aprende a trajetória linear direta entre ruído gaussiano (t=0) e o mel-spectrograma (t=1).
    """
    def __init__(
        self,
        transformer: DiT,
        text_encoder: TextEncoder,
        sigma_min: float = 1e-4,
        cfg_drop_rate: float = 0.15
    ):
        super().__init__()
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.sigma_min = float(sigma_min)
        self.cfg_drop_rate = float(cfg_drop_rate)

    def compute_loss(
        self,
        target_mel: torch.Tensor,
        prompt_mel: torch.Tensor,
        text_tokens: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Calcula a perda de Flow Matching (OT-CFM).
        target_mel: (B, T, mel_dim)
        prompt_mel: (B, T, mel_dim)
        text_tokens: (B, T_text)
        mask: (B, T)
        """
        B, T, mel_dim = target_mel.shape
        device = target_mel.device

        # 1. Amostra timestep t uniformemente em [0, 1]
        t = torch.rand((B,), device=device, dtype=target_mel.dtype)

        # 2. Amostra ruído base x_0 ~ N(0, I)
        x_0 = torch.randn_like(target_mel)
        x_1 = target_mel

        # 3. Interpolação linear do Optimal Transport: x_t = (1 - (1 - sigma_min)*t)*x_0 + t*x_1
        t_expand = t.view(B, 1, 1)
        x_t = (1.0 - (1.0 - self.sigma_min) * t_expand) * x_0 + t_expand * x_1

        # 4. Campo de velocidade alvo u_t = d(x_t)/dt = x_1 - (1 - sigma_min)*x_0
        u_t = x_1 - (1.0 - self.sigma_min) * x_0

        # 5. Codifica o texto e alinha temporalmente com o comprimento do mel
        text_features = self.text_encoder(text_tokens)
        text_features_aligned = self.text_encoder.align_to_mel_length(text_features, target_mel_len=T)

        # Classifier-Free Guidance dropout durante treino
        if self.training and self.cfg_drop_rate > 0:
            drop_mask = torch.rand((B, 1, 1), device=device) < self.cfg_drop_rate
            null_text = self.text_encoder.null_embedding.view(1, 1, -1).expand(B, T, -1)
            text_features_aligned = torch.where(drop_mask, null_text, text_features_aligned)
            prompt_mel = torch.where(drop_mask, torch.zeros_like(prompt_mel), prompt_mel)

        # 6. Predição da velocidade pelo Transformer DiT
        v_pred = self.transformer(
            x_t=x_t,
            t=t,
            prompt_mel=prompt_mel,
            text_features=text_features_aligned,
            mask=mask
        )

        # 7. Cálculo da perda (L1 Loss com mascaramento)
        if mask is not None:
            loss = F.l1_loss(v_pred * mask.unsqueeze(-1), u_t * mask.unsqueeze(-1), reduction="sum")
            denom = mask.sum() * mel_dim + 1e-8
            loss = loss / denom
        else:
            loss = F.l1_loss(v_pred, u_t)

        return {"loss": loss, "l1": loss.detach()}

    @torch.no_grad()
    def sample(
        self,
        prompt_mel: torch.Tensor,
        text_tokens: torch.Tensor,
        target_len: int,
        n_steps: int = 32,
        cfg_strength: float = 2.0,
        solver: str = "euler",
        device: torch.device | str = "cuda"
    ) -> torch.Tensor:
        """
        Gera o espectrograma Mel alvo a partir do prompt de voz e do texto via integração ODE.
        prompt_mel: (1, T_prompt, mel_dim) ou (1, target_len, mel_dim)
        text_tokens: (1, T_text)
        target_len: número de frames do mel de saída
        """
        self.eval()
        B = 1
        mel_dim = self.transformer.mel_dim

        # Prepara prompt mel ajustado ao tamanho alvo
        if prompt_mel.shape[1] < target_len:
            # Se o prompt for mais curto, preenche com zeros no segmento a ser sintetizado
            pad_len = target_len - prompt_mel.shape[1]
            prompt_padded = F.pad(prompt_mel, (0, 0, 0, pad_len))
        else:
            prompt_padded = prompt_mel[:, :target_len, :]

        prompt_padded = prompt_padded.to(device)
        text_tokens = text_tokens.to(device)

        # Codifica texto condicionado
        text_features = self.text_encoder(text_tokens)
        text_features_cond = self.text_encoder.align_to_mel_length(text_features, target_mel_len=target_len)

        # Vetor incondicionado para Classifier-Free Guidance (CFG)
        text_features_uncond = self.text_encoder.null_embedding.view(1, 1, -1).expand(B, target_len, -1).to(device)
        prompt_uncond = torch.zeros_like(prompt_padded)

        # Inicializa com ruído gaussiano x_0 ~ N(0, I)
        x = torch.randn((B, target_len, mel_dim), device=device)

        # Grade temporal de t=0 até t=1
        timesteps = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        dt = 1.0 / n_steps

        # Integração ODE (Euler ou Midpoint)
        for i in range(n_steps):
            t_curr = timesteps[i].repeat(B)

            # Predição condicionada
            v_cond = self.transformer(
                x_t=x,
                t=t_curr,
                prompt_mel=prompt_padded,
                text_features=text_features_cond
            )

            if cfg_strength > 1.0:
                # Predição incondicionada
                v_uncond = self.transformer(
                    x_t=x,
                    t=t_curr,
                    prompt_mel=prompt_uncond,
                    text_features=text_features_uncond
                )
                # Combinação CFG: v = v_uncond + s * (v_cond - v_uncond)
                v = v_uncond + cfg_strength * (v_cond - v_uncond)
            else:
                v = v_cond

            if solver == "midpoint" and i < n_steps - 1:
                # Método de Euler-Heun / Midpoint de 2ª ordem
                x_mid = x + 0.5 * dt * v
                t_mid = t_curr + 0.5 * dt
                v_mid = self.transformer(
                    x_t=x_mid,
                    t=t_mid,
                    prompt_mel=prompt_padded,
                    text_features=text_features_cond
                )
                x = x + dt * v_mid
            else:
                # Método de Euler padrão (1ª ordem - rápido e estável)
                x = x + dt * v

        return x  # (1, target_len, mel_dim)
