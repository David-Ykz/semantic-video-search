import torch
from PIL import Image
from transformers import AutoModel, AutoVideoProcessor

MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"

# use 16 frames instead of 64 frames to speedup inference by ~8x
DEFAULT_NUM_FRAMES = 16

_model = None
_processor = None

def _load_model():
    global _model, _processor
    if _model is None:
        _processor = AutoVideoProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
        _model.eval()
    return _model, _processor


def embed_frames(frames: list[Image.Image]) -> torch.Tensor:
    model, processor = _load_model()
    inputs = processor(frames, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs, skip_predictor=True)
    return output.last_hidden_state.mean(dim=1).squeeze(0)


def _patches_per_temporal_group(model) -> int:
    grid_size = model.config.crop_size // model.config.patch_size
    return grid_size * grid_size


def _patch_indices(model, num_frames: int, start_frame: int, end_frame: int) -> list[int]:
    tubelet_size = model.config.tubelet_size
    patches_per_group = _patches_per_temporal_group(model)

    if num_frames % tubelet_size != 0:
        raise ValueError(f"num_frames ({num_frames}) must be a multiple of tubelet_size ({tubelet_size})")
    if start_frame % tubelet_size != 0 or end_frame % tubelet_size != 0:
        raise ValueError(f"frame range must align to tubelet_size ({tubelet_size})")

    group_start = start_frame // tubelet_size
    group_end = end_frame // tubelet_size
    return list(range(group_start * patches_per_group, group_end * patches_per_group))


def compute_prediction_error(frames: list[Image.Image], target_start_frame: int, target_end_frame: int) -> float:
    """
    Mask out the frames in the sampled interval, and predict its embedding via the window around the interval
    Return the cosine error between the predicted embedding and the entire interval
    A low prediction error implies continuity/coherence in the interval
    """
    model, processor = _load_model()
    inputs = processor(frames, return_tensors="pt")

    num_frames = len(frames)
    target_indices = _patch_indices(model, num_frames, target_start_frame, target_end_frame)
    context_indices = _patch_indices(model, num_frames, 0, target_start_frame) + _patch_indices(
        model, num_frames, target_end_frame, num_frames
    )

    target_mask = [torch.tensor([target_indices], dtype=torch.long)]
    context_mask = [torch.tensor([context_indices], dtype=torch.long)]

    with torch.no_grad():
        output = model(**inputs, context_mask=context_mask, target_mask=target_mask)

    predicted = output.predictor_output.last_hidden_state
    actual = output.predictor_output.target_hidden_state
    return (1 - torch.nn.functional.cosine_similarity(predicted, actual, dim=-1)).mean().item()
