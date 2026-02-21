"""
Loader adaptador para dataset ELSA_D3 em modo streaming.
Consome load_dataset("elsaEU/ELSA_D3", split="train", streaming=True)
e gera (image, label, arch) compatível com o pipeline CoDE.
"""

from typing import Generator, Optional, Tuple, Callable
from PIL import Image
from datasets import load_dataset
import torch
from torch.utils.data import IterableDataset


def _fetch_real_image(url: str, timeout: int = 10, max_retries: int = 2) -> Optional[Image.Image]:
    """Baixa imagem real da URL. Retorna None em caso de falha."""
    try:
        import requests
        from io import BytesIO
        for _ in range(max_retries):
            resp = requests.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            return img
    except Exception:
        pass
    return None


def elsa_streaming_samples(
    max_samples: Optional[int] = None,
    use_real_images: bool = False,
    dataset_name: str = "elsaEU/ELSA_D3",
) -> Generator[Tuple[Image.Image, int, str], None, None]:
    """
    Itera sobre ELSA_D3 em streaming, gerando (image, label, arch).

    Args:
        max_samples: Limite de amostras (imagens) a gerar. None = sem limite.
        use_real_images: Se True, tenta buscar imagens reais via URL (mais lento, pode falhar).
        dataset_name: Nome do dataset no HuggingFace.

    Yields:
        (PIL.Image, label, arch): label 0=Real, 1=Fake; arch="real" ou "gen_0","gen_1",...
    """
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    count = 0

    for element in dataset:
        if max_samples is not None and count >= max_samples:
            break

        # Fakes: image_gen0..3
        for i in range(4):
            if max_samples is not None and count >= max_samples:
                break
            img = element.get(f"image_gen{i}")
            if img is not None:
                if isinstance(img, Image.Image):
                    img = img.convert("RGB")
                elif hasattr(img, "__array__"):
                    import numpy as np
                    img = Image.fromarray(np.asarray(img)).convert("RGB")
                else:
                    continue
                yield img, 1, f"gen_{i}"
                count += 1

        # Reais: fetch da URL (opcional)
        if use_real_images:
            if max_samples is not None and count >= max_samples:
                break
            url = element.get("url")
            if url and isinstance(url, str) and url.strip():
                real_img = _fetch_real_image(url)
                if real_img is not None:
                    yield real_img, 0, "real"
                    count += 1


class ELSAStreamingDataset(IterableDataset):
    """
    Dataset PyTorch iterável que consome ELSA_D3 em streaming.
    Retorna (img_tensor, label, arch) para compatibilidade com validate_d3.
    """

    def __init__(
        self,
        transform: Callable,
        max_samples: Optional[int] = None,
        use_real_images: bool = False,
        dataset_name: str = "elsaEU/ELSA_D3",
    ):
        super().__init__()
        self.transform = transform
        self.max_samples = max_samples
        self.use_real_images = use_real_images
        self.dataset_name = dataset_name

    def __iter__(self):
        for img, label, arch in elsa_streaming_samples(
            max_samples=self.max_samples,
            use_real_images=self.use_real_images,
            dataset_name=self.dataset_name,
        ):
            img_tensor = self.transform(img)
            yield img_tensor, torch.tensor(label, dtype=torch.long), arch
