from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class BuildConfig:
    encoder_id: str = "openai/clip-vit-base-patch32"
    extractor_id: str = "Qwen/Qwen3.5-4B"
    device: str = "cuda"
    max_new_tokens: int = 1536
    text_window: int = 2048
    text_overlap: int = 128
    max_image_side: int = 1024
    top_k: int = 3
    seed: int = 42

    def to_dict(self):
        return asdict(self)

