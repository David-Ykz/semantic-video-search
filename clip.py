import open_clip
import torch
from PIL import Image

_MODEL_NAME = "ViT-B-32"
_PRETRAINED = "laion2b_s34b_b79k"

_TEXT_TEMPLATE = "a photo of {query}."

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

def embed_images(images: list[Image.Image]) -> torch.Tensor:
    model, preprocess, _ = _load_model()
    batch = torch.stack([preprocess(image) for image in images])
    with torch.no_grad():
        embeddings = model.encode_image(batch)
    return embeddings


def embed_image(image: Image.Image) -> torch.Tensor:
    return embed_images([image])[0]


def embed_text(query: str) -> torch.Tensor:
    model, _, tokenizer = _load_model()
    text_input = tokenizer([_TEXT_TEMPLATE.format(query=query)])
    with torch.no_grad():
        embedding = model.encode_text(text_input)
    return embedding.squeeze(0)


def cosine_similarity_scores(
    image_embeddings: torch.Tensor, text_embedding: torch.Tensor
) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(
        image_embeddings, text_embedding.unsqueeze(0)
    )


def cosine_similarity(image: Image.Image, query: str) -> float:
    image_embedding = embed_image(image)
    text_embedding = embed_text(query)
    return cosine_similarity_scores(image_embedding.unsqueeze(0), text_embedding).item()
