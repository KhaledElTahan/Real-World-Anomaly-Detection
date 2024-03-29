"""Dataset Utils"""

import numpy as np
import torch

from src.models import backbone_helper
from src.utils import funcutils


def pack_pathway_output(cfg, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
        cfg: The video model configuration file
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """

    # First, retrieve the backbone configurations file
    backbone_cfg = backbone_helper.get_backbone_merged_cfg(cfg)

    if cfg.DATA.REVERSE_INPUT_CHANNEL:
        frames = frames[[2, 1, 0], :, :, :]
    if backbone_cfg.MODEL.ARCH in backbone_cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frame_list = [frames]
    elif backbone_cfg.MODEL.ARCH in backbone_cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // backbone_cfg.SLOWFAST.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                backbone_cfg.MODEL.ARCH,
                backbone_cfg.MODEL.SINGLE_PATHWAY_ARCH + backbone_cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frame_list


def _frames_to_frames_batches_native(frames :torch.Tensor, batch_size):
    """
    Converts tensor of `channel` x `num frames` x `height` x `width` to list of
    len = `frames / batch size ` of tensors `channel` x `batch size` x `height` x `width`.
    Args:
        frames (torch.Tensor): the frames tensor of format (c, t, h, w)
        batch_size: The size of frames batches
    Returns:
        frames_batches (list(list(torch.Tensor))): The batches of frames
        num_batches (Int): The number of frames batches, i.e. frames/batch_size
    Example:
        _frames_to_frames_batches_native(
            torch.Tensor(3, 65, 240, 320),
            16
        ) =>
        [
            torch.Tensor(3, 16, 240, 320),
            torch.Tensor(3, 16, 240, 320),
            torch.Tensor(3, 16, 240, 320),
            torch.Tensor(3, 16, 240, 320),
            torch.Tensor(3, 1, 240, 320),
        ],
        5
    """
    frames_batches = list(torch.split(frames, batch_size, dim = 1))

    return [frames_batches], len(frames_batches)


def frames_to_frames_batches(cfg, frames):
    """
    Receives list of tensors of frames, either [frames] or [slow_pathway, fast_pathway]
    then converts each frames tensor from `channel` x `num frames` x `height` x `width` to
    list of len = `frames / batch size ` of tensors `channel` x `batch size` x `height` x `width`.
    Args:
        cfg (cfgNode): Video Model Configuration
        frames (list(torch.Tensor)): List of frames from dataset __getitem__
            on the form of [frames] or [slow_pathway, fast_pathway]
    Returns:
        frames_batches list((list(torch.Tensor))): The batches of frames
            on the form of [frames_batches] or [slow_batches, fast_batches]
        num_batches (Int): The number of frames batches, i.e. frames/batch_size
    Example:
        at at cfg.EXTRACT.FRAMES_BATCH_SIZE = 16
        input: ['torch.Size([3, 353, 240, 320])', 'torch.Size([3, 1412, 240, 320])']
        Outout:
        [
            [   # Slow frames
                'torch.Size([3, 4, 240, 320])',
                'torch.Size([3, 4, 240, 320])',
                   ... Same line 84 times ...
                'torch.Size([3, 4, 240, 320])',
                'torch.Size([3, 1, 240, 320])'
            ],
            [   # Fast frames
                'torch.Size([3, 16, 240, 320])',
                'torch.Size([3, 16, 240, 320])',
                   ... Same line 84 times ...
                'torch.Size([3, 16, 240, 320])',
                'torch.Size([3, 4, 240, 320])'
            ]
        ]
    """
    # First, retrieve the backbone configurations file
    backbone_cfg = backbone_helper.get_backbone_merged_cfg(cfg)

    if backbone_cfg.MODEL.ARCH in backbone_cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frames_batches, num_batches = _frames_to_frames_batches_native(frames[0],
            cfg.EXTRACT.FRAMES_BATCH_SIZE)
    elif backbone_cfg.MODEL.ARCH in backbone_cfg.MODEL.MULTI_PATHWAY_ARCH:
        slow_batches, num_slow_batches = _frames_to_frames_batches_native(
            frames[0], int(cfg.EXTRACT.FRAMES_BATCH_SIZE / backbone_cfg.SLOWFAST.ALPHA))
        fast_batches, num_fast_batches = _frames_to_frames_batches_native(frames[1],
            cfg.EXTRACT.FRAMES_BATCH_SIZE)

        # Assume number of fast frames is 257 -> ceil (257/16) = 17
        # Assume SlowFast.Alpha is 4
        # Then, number of slow frames is 64 -> ceil(64/(16/4)) = 16
        if num_fast_batches > num_slow_batches:
            fast_batches[0] = fast_batches[0][:-1]

        frames_batches, num_batches = [slow_batches[0], fast_batches[0]], num_slow_batches
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                backbone_cfg.MODEL.ARCH,
                backbone_cfg.MODEL.SINGLE_PATHWAY_ARCH + backbone_cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frames_batches, num_batches


def _frames_to_batches_of_frames_batches_native(frames, batch_size):
    """
    Given a list of frames batches, stack them into batches of frames batches
    Args:
        frames (list(torch.Tensor)): List of frames batches
        batch_size: batch size of 'frames batch'es
    Returns:
        (list(torch.Tensor)): List of batches of frames batches
    Example:
        _frames_to_batches_of_frames_batches_native(
            [
                torch.Tensor(3, 16, 240, 320),
                torch.Tensor(3, 16, 240, 320),
                torch.Tensor(3, 16, 240, 320),
                torch.Tensor(3, 16, 240, 320),
                torch.Tensor(3, 16, 240, 320),
                torch.Tensor(3, 16, 240, 320),
                torch.Tensor(3, 14, 240, 320),
            ]
            batch_size = 4
        ) =>
        [
            torch.Tensor(4, 3, 16, 240, 320),
            torch.Tensor(2, 3, 16, 240, 320),
            torch.Tensor(1, 3, 14, 240, 320),  # Since frames number is different
        ]
    """
    frames_chunks = [
        frames[idx:idx + batch_size]
        for idx in range(0, len(frames), batch_size)
    ]

    # last frames batch might not be of the same length, hence cannot stack it
    if len(frames_chunks[-1]) > 1 and frames_chunks[-1][-1].shape != frames_chunks[-1][-2].shape:
        last_item = frames_chunks[-1][-1]
        frames_chunks[-1] = frames_chunks[-1][:-1]
        frames_chunks.append([last_item])

    frames_batches = []
    for frames_chunk in frames_chunks:
        frames_batches.append(torch.stack(frames_chunk))

    return frames_batches


def frames_to_batches_of_frames_batches(cfg, frames, drop_last=True):
    """
    Receives list of tensors of frames, either [frames] or [slow_pathway, fast_pathway]
    then converts each frames tensor from `channel` x `num frames` x `height` x `width` to a list of tensors of
    `cfg.EXTRACT.FRAMES_BATCHES_BATCH_SIZE` x `channel` x `cfg.EXTRACT.FRAMES_BATCH_SIZE` x `height` x `width`
    Args:
        cfg (cfgNode): Video Model Configuration
        frames (list(torch.Tensor)): List of frames from dataset __getitem__
            on the form of [frames] or [slow_pathway, fast_pathway]
        drop_last (Bool): drop last batch if frames length is not the same as other batches
    Return:
        frames_batches list((list(torch.Tensor))): The batches of frames
            on the form of [[general_frames_batch], [general_frames_batch], ..] or
            [[slow_batch, fast_batch], [slow_batch, fast_batch], ..]
    Example:
        at cfg.EXTRACT.FRAMES_BATCH_SIZE = 16
        at cfg.EXTRACT.FRAMES_BATCHES_BATCH_SIZE = 8
        input: ['torch.Size([3, 682, 240, 320])', 'torch.Size([3, 2729, 240, 320])']
        output:[
            ['torch.Size([8, 3, 4, 240, 320])', 'torch.Size([8, 3, 16, 240, 320])'],
            ['torch.Size([8, 3, 4, 240, 320])', 'torch.Size([8, 3, 16, 240, 320])'],
                    ......... The same previous line 18 times .........
            ['torch.Size([8, 3, 4, 240, 320])', 'torch.Size([8, 3, 16, 240, 320])'],
            ['torch.Size([2, 3, 4, 240, 320])', 'torch.Size([2, 3, 16, 240, 320])'],
            ['torch.Size([1, 3, 2, 240, 320])', 'torch.Size([1, 3, 9, 240, 320])'] => Will be dropped if drop_last
        ]
    """
    frames, _ = frames_to_frames_batches(cfg, frames)

    backbone_cfg = backbone_helper.get_backbone_merged_cfg(cfg)

    if backbone_cfg.MODEL.ARCH in backbone_cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frames = frames[0]
        general_frames_batches = _frames_to_batches_of_frames_batches_native(
            frames, cfg.EXTRACT.FRAMES_BATCHES_BATCH_SIZE)

        frames_batches = []
        for _, batch in enumerate(general_frames_batches):
            frames_batches.append([batch])

    elif backbone_cfg.MODEL.ARCH in backbone_cfg.MODEL.MULTI_PATHWAY_ARCH:
        slow_frames = frames[0]
        fast_frames = frames[1]
        slow_frames_batches = _frames_to_batches_of_frames_batches_native(
            slow_frames, cfg.EXTRACT.FRAMES_BATCHES_BATCH_SIZE)
        fast_frames_batches = _frames_to_batches_of_frames_batches_native(
            fast_frames, cfg.EXTRACT.FRAMES_BATCHES_BATCH_SIZE)

        frames_batches = []
        for idx, _ in enumerate(slow_frames_batches):
            frames_batches.append(
                [
                    slow_frames_batches[idx],
                    fast_frames_batches[idx]
                ]
            )

    if drop_last and len(frames_batches) > 1 \
        and frames_batches[-1][0].shape[2] != frames_batches[-2][0].shape[2]:
        frames_batches = frames_batches[:-1]

    assert len(frames_batches) > 0
    return frames_batches


def changes_segments_number(tensors_segments, output_segments_num):
    """
    Changes number of input tensors segments
    NOTE: Each new segment will be the mean of many the old segments,
        mapping_indices is used to us which old segments are mapped to the new one
    Args:
        tensors_segments (Torch.Tensor): Tensor with two or more dimensions
        output_segments_num (Int): Number of output segments
    Returns:
        new_segments (Torch.Tensor): Same as tensors_segments except the first
            dimension to be the output_segments_num
        mapping_indices (List): The intervals used to map tensors_segments
            to new_segments
    Examples:
        1)  changes_segments_number(Tensor(50, 2, 3), 5)
            Returns:
                torch.Tensor(5, 2, 3)
                [[0, 9], [10, 19], [20, 28], [29, 38], [39, 49]]
        2)  changes_segments_number(Tensor(2, 2, 3), 4)
            Returns:
                torch.Tensor(4, 2, 3)
                [[0, 0], [0, 0], [0, 0], [1, 1]]
                Note: 3rd Element [0, 0]:
                    1) linspace is [0, 0, 0, 1, 1]
                    1) We don't take last frame to segment
                    2) Because we need it to be next segment's first frame
                    3) No overlapping
    """
    assert isinstance(tensors_segments, torch.Tensor)
    assert len(tensors_segments.shape) >= 2

    # Example => Want to change from 50 segments to 5
    # torch.round(torch.linspace(0, 49, 6))
    # tensor([ 0, 10, 20, 29, 39, 49])
    # 0 -> 9, 10 -> 19, 20 -> 28, 29 -> 38, 39 -> 49
    segments_indices = torch.round(
        torch.linspace(0, tensors_segments.shape[0] - 1, output_segments_num + 1)
        ).to(torch.int).tolist()

    new_shape = (output_segments_num,)
    for dim_value in tensors_segments.shape[1:]:
        new_shape = new_shape + (dim_value,)

    new_segments = torch.zeros(size = new_shape)

    # apply l2 if more than segment?
    mapping_indices = []
    for idx in range(len(segments_indices) - 1):
        ss_idx = segments_indices[idx]
        ee_idx = segments_indices[idx + 1] - 1

        if idx == len(segments_indices) - 2: # Include last frame in last segment
            ee_idx = segments_indices[idx + 1]

        if ee_idx <= ss_idx:
            new_segments[idx] = tensors_segments[ss_idx]
            mapping_indices.append([ss_idx, ss_idx])
        else:
            new_segments[idx] = tensors_segments[ss_idx:ee_idx + 1].mean(0)
            mapping_indices.append([ss_idx, ee_idx])

    return new_segments, mapping_indices


@funcutils.debug(apply=False, sign=True, ret=True, sign_beautify=True, ret_beautify=True)
def segmentize_features(cfg, features_batch, annotation):
    """
    Segmentize features into cfg.EXTRACT.NUMBER_OUTPUT_SEGMENTS number of segments
    and tell whether each segment is anomaly or not
    Args:
        cfg (cfgNode): Video model configurations node
        features_batch (torch.Tensor): Features of a whole video
        annotation (Tuple): Frames intervals in which the video is anomalous
    Returns:
        features_segments (torch.Tensor): cfg.EXTRACT.NUMBER_OUTPUT_SEGMENTS Segments of frames
        is_anomaly_segment (torch.Tensor): tells us whether each segment is anomalous or not
    """

    features_segments, mapping_indices = changes_segments_number(features_batch,
        cfg.EXTRACT.NUMBER_OUTPUT_SEGMENTS)

    is_anomaly_segment = torch.zeros(size = (cfg.EXTRACT.NUMBER_OUTPUT_SEGMENTS,), dtype=int)

    def _index_to_frame(index, frames_batch_size, start):
        frame = index * frames_batch_size
        if not start:
            frame += frames_batch_size - 1
        return frame

    mapping_frames = []
    for indices in mapping_indices:
        mapping_frames.append(
            [
                _index_to_frame(indices[0], cfg.EXTRACT.FRAMES_BATCH_SIZE, True),
                _index_to_frame(indices[1], cfg.EXTRACT.FRAMES_BATCH_SIZE, False)
            ]
        )

    def _is_inside(start_frame, end_frame, segment_start_frame, segment_end_frame):
        assert start_frame < end_frame
        assert segment_start_frame < segment_end_frame

        if start_frame > segment_end_frame:
            return False
        if end_frame < segment_start_frame:
            return False
        return True

    idx = 0
    while idx < len(annotation) and annotation[idx] != -1:
        start_frame = annotation[idx]
        end_frame = annotation[idx + 1]

        for i, segment_frames in enumerate(mapping_frames):
            if _is_inside(start_frame, end_frame, segment_frames[0], segment_frames[1]):
                is_anomaly_segment[i] = 1

        idx += 2

    return features_segments, is_anomaly_segment


def as_binary_vector(labels, num_classes):
    """
    Construct binary label vector given a list of label indices.
    Args:
        labels (list): The input label list.
        num_classes (int): Number of classes of the label vector.
    Returns:
        labels (numpy array): the resulting binary vector.
    """
    label_arr = np.zeros((num_classes,))

    for lbl in set(labels):
        label_arr[lbl] = 1.0
    return label_arr


def label_to_one_hot(label, sorted_classes_list):
    """
    Construct ont hot vector given a label
    Args:
        label (str): The input label.
        sorted_classes_list (List): Sorted list of all classes in dataset
    Returns:
        one_hot (Torch.Tesnor): One hot encoding of the label
    """
    one_hot = torch.zeros(len(sorted_classes_list), dtype=torch.int64)
    one_hot[sorted_classes_list.index(label)] = 1

    return one_hot
