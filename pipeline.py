import queue
import threading

import cv2
import torch
from PIL import Image

from clip import cosine_similarity_scores, embed_images, embed_text

DEFAULT_SAMPLING_INTERVAL = 2
DEFAULT_BATCH_SIZE = 16
DEFAULT_THRESHOLD = 0.25
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
    batch_size: int = DEFAULT_BATCH_SIZE,
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

def score_query(timestamps: list[float], embeddings: torch.Tensor, query: str):
    text_embedding = embed_text(query)
    scores = cosine_similarity_scores(embeddings, text_embedding)
    return list(zip(timestamps, scores.tolist()))

def matches_above_threshold(scored: list[tuple[float, float]], threshold: float):
    return [(timestamp, score) for timestamp, score in scored if score >= threshold]

def search_video(
    video_path: str,
    query: str,
    sampling_interval: float = DEFAULT_SAMPLING_INTERVAL,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    timestamps, embeddings = embed_video(video_path, sampling_interval, batch_size)
    scored = score_query(timestamps, embeddings, query)
    matches = matches_above_threshold(scored, threshold)
    return matches, scored, embeddings
