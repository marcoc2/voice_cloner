"""
Compatibilidade com o huggingface_hub 1.x.

Várias bibliotecas de áudio ainda usam a API antiga do hub. Não dá para voltar a
versão: o `transformers` 5.x, que roda o Whisper deste projeto, exige
`huggingface-hub>=1.5`. Também não dá para atualizar essas bibliotecas — as
versões novas exigem `torch>=2.8`, o que trocaria o torch+cu124 por um build sem
CUDA e reintroduziria o `torchcodec` (ver problemas #2 e #11 do histórico).

Então traduzimos a API antiga para a nova aqui, num lugar só.
"""
import functools
import importlib
import sys

# Argumentos aceitos pelo hf_hub_download antigo e removidos no hub 1.x.
_ARGS_REMOVIDOS = ("proxies", "resume_download", "legacy_cache_layout", "local_dir_use_symlinks")

# Módulos que importam hf_hub_download direto (`from huggingface_hub import ...`)
# e por isso guardam a própria referência à função original.
_MODULOS_COM_COPIA = (
    "pyannote.audio.core.model",
    "pyannote.audio.core.pipeline",
    "pyannote.audio.pipelines.speaker_verification",
    "seed_vc.modules.bigvgan.bigvgan",
    "seed_vc.hf_utils",
)


def patch_hf_hub_download():
    """
    Aceita a assinatura antiga: traduz `use_auth_token` para `token` e descarta
    os argumentos que o hub 1.x não conhece mais.
    """
    import huggingface_hub

    original = huggingface_hub.hf_hub_download
    if getattr(original, "_compat_aplicado", False):
        shim = original
    else:
        @functools.wraps(original)
        def shim(*args, **kwargs):
            legado = kwargs.pop("use_auth_token", None)
            if legado is not None and kwargs.get("token") is None:
                kwargs["token"] = legado
            for nome in _ARGS_REMOVIDOS:
                kwargs.pop(nome, None)
            return original(*args, **kwargs)

        shim._compat_aplicado = True
        huggingface_hub.hf_hub_download = shim

    for nome_mod in _MODULOS_COM_COPIA:
        modulo = sys.modules.get(nome_mod)
        if modulo is None:
            try:
                modulo = importlib.import_module(nome_mod)
            except Exception:
                continue
        if hasattr(modulo, "hf_hub_download"):
            modulo.hf_hub_download = shim


def patch_bigvgan():
    """
    O BigVGAN embutido no seed-vc declara `proxies` e `resume_download` como
    obrigatórios (keyword-only), mas o hub 1.x parou de passá-los — o carregamento
    morre com TypeError antes mesmo de baixar os pesos. Preenchemos os dois.
    """
    try:
        from seed_vc.modules.bigvgan.bigvgan import BigVGAN
    except Exception:
        return

    atual = BigVGAN.__dict__.get("_from_pretrained")
    if atual is None or getattr(atual, "_compat_aplicado", False):
        return

    original = atual.__func__ if isinstance(atual, classmethod) else atual
    if getattr(original, "_compat_aplicado", False):
        return

    @functools.wraps(original)
    def shim(cls, **kwargs):
        kwargs.setdefault("proxies", None)
        kwargs.setdefault("resume_download", False)
        return original(cls, **kwargs)

    shim._compat_aplicado = True
    BigVGAN._from_pretrained = classmethod(shim)


def patch_torch_load():
    """
    O torch 2.6 passou a usar `weights_only=True` por padrão em `torch.load`, e
    os checkpoints Lightning do pyannote guardam objetos fora da lista branca.

    Em vez de desligar a verificação (`weights_only=False`, que executaria pickle
    arbitrário), liberamos só as classes que esses checkpoints usam: uma do torch
    e três do próprio pyannote.
    """
    import torch

    permitidos = []
    try:
        from torch.torch_version import TorchVersion
        permitidos.append(TorchVersion)
    except Exception:
        pass
    try:
        from pyannote.audio.core.task import Problem, Resolution, Specifications
        permitidos.extend([Specifications, Problem, Resolution])
    except Exception:
        pass

    if permitidos:
        torch.serialization.add_safe_globals(permitidos)


def apply_all():
    """Aplica todos os remendos. Idempotente."""
    patch_hf_hub_download()
    patch_torch_load()
    patch_bigvgan()
