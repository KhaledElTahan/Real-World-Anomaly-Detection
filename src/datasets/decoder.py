"""Decode video into frames"""
import math
import random
import numpy as np
import torch
import torchvision.io as io

from src.utils import funcutils

def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames


def pyav_decode_stream(
    container, start_pts, end_pts, stream, stream_name, buffer_size=0
):
    """
    Decode the video with PyAV decoder.
    Args:
        container (container): PyAV container.
        start_pts (int): the starting Presentation TimeStamp to fetch the
            video frames.
        end_pts (int): the ending Presentation TimeStamp of the decoded frames.
        stream (stream): PyAV stream.
        stream_name (dict): a dictionary of streams. For example, {"video": 0}
            means video stream at stream index 0.
        buffer_size (int): number of additional frames to decode beyond end_pts.
    Returns:
        result (list): list of frames decoded.
        max_pts (int): max Presentation TimeStamp of the video sequence.
    """
    # Seeking in the stream is imprecise. Thus, seek to an ealier PTS by a
    # margin pts.
    margin = 1024
    seek_offset = max(start_pts - margin, 0)

    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    frames = {}
    buffer_count = 0
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            frames[frame.pts] = frame
        else:
            buffer_count += 1
            frames[frame.pts] = frame
            if buffer_count >= buffer_size:
                break
    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts


@funcutils.force_garbage_collection(before=True, after=True)
def pyav_decode(container):
    """
    Decode the video into frames
    Args:
        container (container): pyav container.
    Returns:
        frames (tensor): decoded frames from the video.
        fps (float): the number of frames per second of the video.
    """
    fps = float(container.streams.video[0].average_rate)
    video_start_pts, video_end_pts = 0, math.inf

    # Check if video stream was found
    assert container.streams.video

    video_frames, _ = pyav_decode_stream(
        container,
        video_start_pts,
        video_end_pts,
        container.streams.video[0],
        {"video": 0},
    )
    container.close()

    frames = torch.as_tensor(np.stack([frame.to_rgb().to_ndarray() for frame in video_frames]))

    return frames, fps


def decode(container, backend="pyav"):
    """
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        backend (str): decoding backend.
    Returns:
        frames (tensor): decoded frames from the video.
    """

    try:
        if backend == "pyav":
            frames, fps = pyav_decode(container)
        else:
            raise NotImplementedError("Unknown decoding backend {}".format(backend))
    except Exception as exc:
        print("Failed to decode by {} with exception: {}".format(backend, exc))
        return None

    # Return None if the frames was not decoded successfully.
    if frames is None or frames.size(0) == 0:
        return None

    return frames
