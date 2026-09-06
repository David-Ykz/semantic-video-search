import queue
import statistics
import threading

import cv2
import torch
from PIL import Image

from clip import cosine_similarity_scores, embed_images, embed_text

DEFAULT_SAMPLING_INTERVAL = 2
CLIP_BATCH_SIZE = 16

MIN_CLIP_SIMILARITY_THRESHOLD = 0.22
CLIP_THRESHOLD_STDDEV_MULTIPLIER = 2.0

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

# Use a dynamic threshold based on the distribution of similarity scores for a video
def compute_similarity_threshold(
    scores: list[float],
    stddev_multiplier: float = CLIP_THRESHOLD_STDDEV_MULTIPLIER,
    min_threshold: float = MIN_CLIP_SIMILARITY_THRESHOLD,
) -> float:
    mean = statistics.fmean(scores)
    stdev = statistics.pstdev(scores)
    return max(min_threshold, mean + stddev_multiplier * stdev)

def search_video(
    video_path: str,
    query: str,
    sampling_interval: float = DEFAULT_SAMPLING_INTERVAL,
    clip_batch_size: int = CLIP_BATCH_SIZE,
    stddev_multiplier: float = CLIP_THRESHOLD_STDDEV_MULTIPLIER,
    min_threshold: float = MIN_CLIP_SIMILARITY_THRESHOLD,
):
    timestamps, embeddings = embed_video(video_path, sampling_interval, clip_batch_size)
    scored = score_query_similarity(timestamps, embeddings, query)
    threshold = compute_similarity_threshold([score for _, score in scored], stddev_multiplier, min_threshold)
    matches = filter_clip_scores(scored, threshold)
    return matches, threshold, scored, embeddings
