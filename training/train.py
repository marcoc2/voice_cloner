import os
import sys
import copy
import argparse
from pathlib import Path

# Garante acesso aos módulos do projeto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models.dit import DiT
from models.text_encoder import TextEncoder
from models.flow_matching import ConditionalFlowMatching
from dataset.audio_dataset import VoiceCloningDataset, voice_cloning_collate_fn


class EMA:
    """
    Exponential Moving Average (EMA) para manter uma média ponderada dos pesos do modelo.
    Melhora significativamente a estabilidade e qualidade auditiva das amostras geradas.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}


class Trainer:
    """
    Pipeline de treinamento e Fine-Tuning para o Clonador de Voz Moderno.
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Dispositivo de treinamento: {self.device}")

        # Instanciação do Modelo
        m_cfg = config["model"]
        self.dit = DiT(
            dim=m_cfg["dim"],
            depth=m_cfg["depth"],
            heads=m_cfg["heads"],
            dim_head=m_cfg["dim_head"],
            ff_mult=m_cfg["ff_mult"],
            mel_dim=m_cfg["mel_dim"],
            text_dim=m_cfg["text_dim"],
            dropout=m_cfg.get("dropout", 0.1)
        )
        self.text_encoder = TextEncoder(
            vocab_size=m_cfg.get("vocab_size", 256),
            embed_dim=m_cfg["text_dim"],
            depth=4
        )
        self.cfm = ConditionalFlowMatching(
            transformer=self.dit,
            text_encoder=self.text_encoder,
            sigma_min=m_cfg.get("sigma_min", 1e-4)
        ).to(self.device)

        # EMA e Otimizador
        t_cfg = config["training"]
        self.ema = EMA(self.cfm, decay=t_cfg.get("ema_decay", 0.999))
        self.optimizer = torch.optim.AdamW(
            self.cfm.parameters(),
            lr=float(t_cfg["learning_rate"]),
            weight_decay=float(t_cfg.get("weight_decay", 1e-4)),
            betas=(0.9, 0.99)
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

        # Diretórios de saída
        self.checkpoint_dir = Path(t_cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = 0

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.cfm.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Época {epoch + 1}/{self.config['training']['max_epochs']}")

        for step, batch in enumerate(pbar):
            target_mel = batch["target_mel"].to(self.device)
            prompt_mel = batch["prompt_mel"].to(self.device)
            text_tokens = batch["text_tokens"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                loss_dict = self.cfm.compute_loss(
                    target_mel=target_mel,
                    prompt_mel=prompt_mel,
                    text_tokens=text_tokens,
                    mask=mask
                )
                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.cfm.parameters(),
                max_norm=self.config["training"].get("grad_clip", 1.0)
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Atualiza pesos na média exponencial móvel (EMA)
            self.ema.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(1, len(dataloader))

    def save_checkpoint(self, epoch: int, is_best: bool = False, filename: str | None = None):
        if filename is None:
            filename = f"checkpoint_epoch_{epoch + 1}.pt" if not is_best else "best_model.pt"
        save_path = self.checkpoint_dir / filename

        state = {
            "epoch": epoch + 1,
            "state_dict": self.cfm.state_dict(),
            "ema_shadow": self.ema.shadow,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config
        }
        torch.save(state, save_path)
        print(f"[Trainer] Checkpoint salvo em: {save_path}")

    def load_checkpoint(self, checkpoint_path: str | Path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"[Trainer] Checkpoint {checkpoint_path} não encontrado!")
            return

        print(f"[Trainer] Carregando checkpoint de: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        self.cfm.load_state_dict(state["state_dict"])
        if "ema_shadow" in state:
            self.ema.shadow = state["ema_shadow"]
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "epoch" in state:
            self.start_epoch = state["epoch"]
        print(f"[Trainer] Checkpoint carregado com sucesso (iniciando da época {self.start_epoch})!")

    def fit(self, data_dir: str | Path, metadata_file: str | Path | None = None):
        dataset = VoiceCloningDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            sample_rate=self.config["audio"]["sample_rate"]
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            collate_fn=voice_cloning_collate_fn,
            num_workers=0,
            pin_memory=(self.device.type == "cuda")
        )

        max_epochs = self.config["training"]["max_epochs"]
        save_every = self.config["training"].get("save_every_epochs", 5)

        print(f"[Trainer] Iniciando treinamento por {max_epochs} épocas...")
        for epoch in range(self.start_epoch, max_epochs):
            avg_loss = self.train_epoch(dataloader, epoch)
            print(f"[Trainer] Época {epoch + 1}/{max_epochs} concluída - Loss Média: {avg_loss:.4f}")

            if (epoch + 1) % save_every == 0 or (epoch + 1) == max_epochs:
                self.save_checkpoint(epoch)


def main():
    parser = argparse.ArgumentParser(description="Treinador do Clonador de Voz Moderno (Flow Matching DiT)")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Caminho para o arquivo YAML de configuração")
    parser.add_argument("--data_dir", type=str, required=True, help="Diretório contendo os áudios para treino/fine-tuning")
    parser.add_argument("--metadata", type=str, default=None, help="Arquivo de metadados opcional (.json ou .csv)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Caminho para checkpoint prévio para continuar treino")
    parser.add_argument("--epochs", type=int, default=None, help="Sobrescrever número de épocas")
    parser.add_argument("--batch_size", type=int, default=None, help="Sobrescrever tamanho do lote")
    parser.add_argument("--lr", type=float, default=None, help="Sobrescrever taxa de aprendizado")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["max_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    trainer = Trainer(config)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    trainer.fit(data_dir=args.data_dir, metadata_file=args.metadata)


if __name__ == "__main__":
    main()
