import queue
import threading

import cv2
import torch
from PIL import Image

from clip import cosine_similarity_scores, embed_images, embed_text
from jepa import compute_prediction_error

DEFAULT_SAMPLING_INTERVAL = 2
CLIP_BATCH_SIZE = 16
CLIP_SIMILARITY_THRESHOLD = 0.25
TARGET_INTERVAL_WIDTH = 2.0
CONTEXT_INTERVAL_WIDTH = 2.0
JEPA_ERROR_THRESHOLD = 0.5

END_OF_VIDEO = object()

def sample_frames(video_path: str, sampling_interval: float = DEFAULT_SAMPLING_INTERVAL):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise ValueError(f"could not read FPS for video: {video_path}")

    frame_interval = max(1, round(sampling_interval * fps))

    try:
        frame_index = 0
        while True:
            if frame_index % frame_interval == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_index / fps, Image.fromarray(rgb)
            else:
                ok = cap.grab()
                if not ok:
                    break
            frame_index += 1
    finally:
        cap.release()

def _batched(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch

def embed_video(
    video_path: str,
    sampling_interval: float = DEFAULT_SAMPLING_INTERVAL,
    batch_size: int = CLIP_BATCH_SIZE,
):
    frame_batches: queue.Queue = queue.Queue(maxsize=4)

    def produce():
        try:
            for batch in _batched(sample_frames(video_path, sampling_interval), batch_size):
                frame_batches.put(batch)
        finally:
            frame_batches.put(END_OF_VIDEO)

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()

    timestamps = []
    embedding_batches = []
    while True:
        batch = frame_batches.get()
        if batch is END_OF_VIDEO:
            break
        timestamps.extend(timestamp for timestamp, _ in batch)
        embedding_batches.append(embed_images([frame for _, frame in batch]))

    producer.join()

    if not embedding_batches:
        raise ValueError(f"no frames sampled from video: {video_path}")

    return timestamps, torch.cat(embedding_batches, dim=0)

def score_query_similarity(timestamps: list[float], embeddings: torch.Tensor, query: str):
    text_embedding = embed_text(query)
    scores = cosine_similarity_scores(embeddings, text_embedding)
    return list(zip(timestamps, scores.tolist()))

def filter_clip_scores(scored: list[tuple[float, float]], threshold: float):
    return [(timestamp, score) for timestamp, score in scored if score >= threshold]

def score_coherence(
    video_path: str,
    timestamp: float,
    target_width: float = TARGET_INTERVAL_WIDTH,
    context_width: float = CONTEXT_INTERVAL_WIDTH,
) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    window_start = timestamp - target_width / 2 - context_width
    window_end = timestamp + target_width / 2 + context_width

    cap.set(cv2.CAP_PROP_POS_FRAMES, round(window_start * fps))
    num_frames = round((window_end - window_start) * fps)
    frames = []
    for _ in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()

    target_start_frame = round(context_width * fps)
    target_end_frame = target_start_frame + round(target_width * fps)
    return compute_prediction_error(frames, target_start_frame, target_end_frame)

def filter_coherence_scores(
    video_path: str,
    matches: list[tuple[float, float]],
    target_width: float = TARGET_INTERVAL_WIDTH,
    context_width: float = CONTEXT_INTERVAL_WIDTH,
    error_threshold: float = JEPA_ERROR_THRESHOLD,
):
    scored = [
        (timestamp, clip_score, score_coherence(video_path, timestamp, target_width, context_width))
        for timestamp, clip_score in matches
    ]
    accepted = [(timestamp, clip_score, error) for timestamp, clip_score, error in scored if error <= error_threshold]
    return accepted, scored

def search_video(
    video_path: str,
    query: str,
    sampling_interval: float = DEFAULT_SAMPLING_INTERVAL,
    clip_threshold: float = CLIP_SIMILARITY_THRESHOLD,
    clip_batch_size: int = CLIP_BATCH_SIZE,
    target_width: float = TARGET_INTERVAL_WIDTH,
    context_width: float = CONTEXT_INTERVAL_WIDTH,
    jepa_error_threshold: float = JEPA_ERROR_THRESHOLD,
):
    timestamps, embeddings = embed_video(video_path, sampling_interval, clip_batch_size)
    scored = score_query_similarity(timestamps, embeddings, query)
    clip_matches = filter_clip_scores(scored, clip_threshold)
    matches, coherence_scored = filter_coherence_scores(
        video_path, clip_matches, target_width, context_width, jepa_error_threshold
    )
    return matches, coherence_scored, clip_matches, scored, embeddings
