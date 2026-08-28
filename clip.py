import open_clip
import torch
from PIL import Image

_MODEL_NAME = "ViT-B-32-quickgelu"
_PRETRAINED = "openai"

_model = None
_preprocess = None
_tokenizer = None

def _load_model():
    global _model, _preprocess, _tokenizer
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            _MODEL_NAME, pretrained=_PRETRAINED
        )
        _model.eval()
        _tokenizer = open_clip.get_tokenizer(_MODEL_NAME)
    return _model, _preprocess, _tokenizer

def embed_image(image: Image.Image) -> torch.Tensor:
    model, preprocess, _ = _load_model()
    image_input = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(image_input)
    return embedding.squeeze(0)


def embed_text(query: str) -> torch.Tensor:
    model, _, tokenizer = _load_model()
    text_input = tokenizer([query])
    with torch.no_grad():
        embedding = model.encode_text(text_input)
    return embedding.squeeze(0)


def cosine_similarity(image: Image.Image, query: str) -> float:
    image_embedding = embed_image(image)
    text_embedding = embed_text(query)
    return torch.nn.functional.cosine_similarity(
        image_embedding.unsqueeze(0), text_embedding.unsqueeze(0)
    ).item()
