"""DiDeMo dataset loader for video-text retrieval.

DiDeMo (Distinct Describing Moments) is an unedited video dataset with
dense temporal annotations. Each video is paired with multiple natural
language descriptions, where each description is localized to a specific
start and end time. Fine-grained temporal descriptions from the VILA JSON
are split into sentences to support moment-level alignment learning.

Sample return format (training):
    (input_ids, input_mask,                          # aggregated global caption token/mask
     input_list_ids, input_list_mask,                 # sentence-level token IDs and masks
     input_list_word_mask,                            # word-level attention masks
     video, video_mask,                               # video frames / frame mask
     feature_idx, hash(video_id))

Expected data layout:
    data_path/
        train_data.json / val_data.json / test_data.json  - temporal annotations + time ranges
        train_list.txt  / val_list.txt  / test_list.txt   - video ID lists
        DiDeMo_VILA_F6.json  - sentence-level temporal descriptions
    features_path/
        {video_id}.mp4
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import json

from .rawvideo_util import RawVideoExtractor


class DiDeMoDataset(Dataset):
    """DiDeMo dataset loader.

    Loads temporal annotations, aggregates them into per-video caption groups,
    and associates each video with fine-grained sentence descriptions via
    DiDeMo_VILA_F6.json.
    """

    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=2,
    ):
        """Initialize the DiDeMo dataset loader.

        Args:
            subset: One of 'train', 'val', 'test'.
            data_path: Root directory with JSON annotations and video ID lists.
            features_path: Root directory containing MP4 video files.
            tokenizer: Text tokenizer (e.g., CLIPTokenizer).
            max_words: Max token count per text (including CLS/SEP); longer texts truncated.
            feature_framerate: Sampling FPS (default 1.0 = 1 frame/sec).
            max_frames: Max frames per video; exceeding clips are truncated.
            image_resolution: Frame spatial resolution (default 224).
            frame_order: Temporal ordering strategy: 0=original / 1=reverse / 2=random.
            slice_framepos: Frame sampling strategy: 0=front / 1=back / 2=uniform.
        """
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2], "frame_order must be 0/1/2"
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2], "slice_framepos must be 0/1/2"

        self.subset = subset
        assert self.subset in ["train", "val", "test"], "subset must be train/val/test"

        # ---- Load per-subset video ID lists ----
        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, "train_list.txt")
        video_id_path_dict["val"] = os.path.join(self.data_path, "val_list.txt")
        video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")

        # ---- Load temporal annotation JSONs ----
        video_json_path_dict = {}
        video_json_path_dict["train"] = os.path.join(self.data_path, "train_data.json")
        video_json_path_dict["val"] = os.path.join(self.data_path, "val_data.json")
        video_json_path_dict["test"] = os.path.join(self.data_path, "test_data.json")

        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]

        # ---- Load fine-grained VILA temporal descriptions ----
        caps_data_path = os.path.join(self.data_path, "DiDeMo_VILA_F6.json")
        with open(caps_data_path, 'r') as f:
            caps_data = json.load(f)

        # ---- Pre-build index: name -> cap_item for O(1) lookup ----
        cap_dict = {item['name']: item for item in caps_data}

        # ---- Build caption dictionary ----
        caption_dict = {}
        with open(video_json_path_dict[self.subset], 'r') as f:
            json_data = json.load(f)
        
        # ---- Target sentence count after padding (same as Charades) ----
        caps_len = 10

        for itm in json_data:
            description = itm["description"]
            times = itm["times"]
            video = itm["video"]
            if video not in video_ids:
                continue
            # Compute average start/end time across all temporal annotations
            start_ = np.mean([t_[0] for t_ in times]) * 5
            end_ = (np.mean([t_[1] for t_ in times]) + 1) * 5
            cap_item = cap_dict.get(video)
            if cap_item is not None:
                cap_list = [s.strip() for s in cap_item['description'].split(".") if s.strip()]
                num_real = len(cap_list)
                # Pad to fixed length by repeating the last fragment
                if num_real < caps_len:
                    cap_list.extend([cap_list[-1]] * (caps_len - num_real))
                cap_list = cap_list[:caps_len]
                # Binary mask: 1 for genuine sentences, 0 for padded
                cap_mask = [1] * num_real + [0] * (caps_len - num_real)
                cap_mask = cap_mask[:caps_len]
                if video in caption_dict:
                    caption_dict[video]["start"].append(start_)
                    caption_dict[video]["end"].append(end_)
                    caption_dict[video]["text"].append(description)
                    caption_dict[video]["cap_list"].append(cap_list)
                    caption_dict[video]["cap_mask"].append(cap_mask)
                else:
                    caption_dict[video] = {}
                    caption_dict[video]["start"] = [start_]
                    caption_dict[video]["end"] = [end_]
                    caption_dict[video]["text"] = [description]
                    caption_dict[video]["cap_list"].append(cap_list)
                    caption_dict[video]["cap_mask"].append(cap_mask)

        # Normalize temporal ranges to full video duration [0, 31]
        for k_ in caption_dict.keys():
            caption_dict[k_]["start"] = [0]
            caption_dict[k_]["end"] = [31]
            caption_dict[k_]["text"] = [" ".join(caption_dict[k_]["text"])]

        # ---- Scan video directory ----
        video_dict = {}
        for root, dub_dir, video_files in os.walk(self.features_path):
            for video_file in video_files:
                video_id_ = os.path.splitext(video_file)[0]
                if video_id_ not in video_ids:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_

        self.caption_dict = caption_dict
        self.video_dict = video_dict
        # Retain only videos that have both captions and video files
        video_ids = list(set(video_ids) & set(self.caption_dict.keys()) & set(self.video_dict.keys()))

        # ---- Build iteration index: (video_id, sub_id) pairs ----
        self.iter2video_pairs_dict = {}
        for video_id in self.caption_dict.keys():
            if video_id not in video_ids:
                continue
            caption = self.caption_dict[video_id]
            n_caption = len(caption['start'])
            for sub_id in range(n_caption):
                self.iter2video_pairs_dict[len(self.iter2video_pairs_dict)] = (video_id, sub_id)

        # ---- Video frame extractor (OpenCV backend) ----
        self.rawVideoExtractor = RawVideoExtractor(
            framerate=feature_framerate, size=image_resolution)

        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "<|unk|>",
            "PAD_TOKEN": "<|pad|>",
        }

    def __len__(self):
        """Number of (video, caption) iteration pairs."""
        return len(self.iter2video_pairs_dict)

    def _get_text(self, video_id, sub_id):
        """Convert text annotations into model-ready token/mask tensors.

        Produces two parallel representations:
            1. A single aggregated text description for the full video.
            2. A list of independently tokenized sentence-level texts.
        Both are padded with CLS/SEP to exactly max_words tokens.

        Args:
            video_id: Video identifier.
            sub_id: Caption subset index within the video.
        Returns:
            input_ids          : shape (1, max_words), aggregated caption token IDs.
            input_mask         : shape (1, max_words), aggregated attention mask.
            input_list_ids     : list of sentence-level (1, max_words) token IDs.
            input_list_mask    : list of sentence-level (1, max_words) caption masks.
            input_list_word_mask: list of sentence-level (1, max_words) word masks.
            starts             : temporal start frame index (np.int64 array).
            ends               : temporal end frame index (np.int64 array).
        """

        # region ---- Part 1: Single aggregated global caption ----
        caption = self.caption_dict[video_id]
        k = 1
        r_ind = [sub_id]

        starts = np.zeros(k, dtype=np.int64)
        ends = np.zeros(k, dtype=np.int64)

        ind = r_ind[0]
        start_, end_ = caption['start'][ind], caption['end'][ind]
        words = self.tokenizer.tokenize(caption['text'][ind])
        starts[0], ends[0] = start_, end_

        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words

        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)

        # region ---- Part 2: Sentence-level fine-grained texts ----
        input_list_ids, input_list_mask, input_list_word_mask = [], [], []

        caption = self.caption_dict[video_id]
        ind = sub_id
        start_, end_ = caption['start'][ind], caption['end'][ind]
        caption, caption_mask = caption['cap_list'][ind], caption['cap_mask'][ind]
        input_list_mask.append(caption_mask)

        # Tokenize only valid sentences (where cap_mask == 1)
        for caption_, caption_mask_ in zip(caption, caption_mask):
            if not caption_mask_:
                continue
            words = self.tokenizer.tokenize(caption_)

            # Prepend CLS, append SEP, truncate if needed, pad to max_words
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids_ = self.tokenizer.convert_tokens_to_ids(words)
            input_mask_ = [1] * len(input_ids_)
            while len(input_ids_) < self.max_words:
                input_ids_.append(0)
                input_mask_.append(0)

            assert len(input_ids_) == self.max_words
            assert len(input_mask_) == self.max_words

            input_ids_ = np.array(input_ids_)
            input_mask_ = np.array(input_mask_)

            input_list_ids.append(input_ids_)
            input_list_word_mask.append(input_mask_)

        return (input_ids, input_mask, 
                input_list_ids, input_list_mask, input_list_word_mask,
                starts, ends)

    def _get_rawvideo(self, idx, s, e):
        """Extract and preprocess video frames for a given time range.

        Args:
            idx: Video identifier (key in self.video_dict).
            s: Start frame indices (used to slice into the video).
            e: End frame indices (used to slice into the video).
        Returns:
            video      : shape (len(s), max_frames, 1, 3, H, W) video tensor.
            video_mask : shape (max_frames,) binary mask of valid frames.

        Time range handling:
            - Negative values clamped to 0.
            - If start > end, swap them.
            - If start == end, add 1 frame padding to avoid empty clips.

        Frame sampling strategy (slice_framepos):
            0 -> keep the first max_frames;
            1 -> keep the last max_frames;
            2 -> uniformly sample max_frames across the full clip.
        """
        video_mask = np.zeros(self.max_frames, dtype=np.int64)
        max_video_length = 0

        # Pre-allocate: (num_clips, max_frames, 1, C, H, W)
        video = np.zeros(
            (len(s), self.max_frames, 1, 3,
             self.rawVideoExtractor.size, self.rawVideoExtractor.size),
            dtype=float
        )
        video_path = self.video_dict[idx]

        try:
            for i in range(len(s)):
                # Normalize time range boundaries
                start_time = int(s[i])
                end_time = int(e[i])
                start_time = start_time if start_time >= 0. else 0.
                end_time = end_time if end_time >= 0. else 0.
                if start_time > end_time:
                    start_time, end_time = end_time, start_time
                elif start_time == end_time:
                    end_time = end_time + 1  # avoid zero-length clips

                cache_id = "{}_{}_{}".format(video_path, start_time, end_time)
                raw_video_data = self.rawVideoExtractor.get_video_data(
                    video_path, start_time, end_time)
                raw_video_data = raw_video_data['video']

                if len(raw_video_data.shape) > 3:
                    raw_video_data_clip = raw_video_data
                    raw_video_slice = self.rawVideoExtractor.process_raw_data(
                        raw_video_data_clip)

                    # ---- Frame truncation / uniform sampling ----
                    if self.max_frames < raw_video_slice.shape[0]:
                        if self.slice_framepos == 0:
                            video_slice = raw_video_slice[:self.max_frames, ...]
                        elif self.slice_framepos == 1:
                            video_slice = raw_video_slice[-self.max_frames:, ...]
                        else:
                            sample_indx = np.linspace(
                                0, raw_video_slice.shape[0] - 1,
                                num=self.max_frames, dtype=int)
                            video_slice = raw_video_slice[sample_indx, ...]
                    else:
                        video_slice = raw_video_slice

                    video_slice = self.rawVideoExtractor.process_frame_order(
                        video_slice, frame_order=self.frame_order)

                    slice_len = video_slice.shape[0]
                    max_video_length = max_video_length if max_video_length > slice_len else slice_len
                    if slice_len >= 1:
                        video[i][:slice_len, ...] = video_slice
                else:
                    print("video path: {} error. video id: {}, start: {}, end: {}".format(
                        video_path, idx, start_time, end_time))
        except Exception as excep:
            print("video path: {} error. video id: {}, start: {}, end: {}, Error: {}".format(
                video_path, idx, s, e, excep))

        video_mask[:max_video_length] = [1] * max_video_length

        return video, video_mask

    def __getitem__(self, feature_idx):
        """Return a single training sample.

        Returns a full sample tuple containing:
            - global aggregated caption (input_ids, input_mask)
            - sentence-level captions (input_list_ids, input_list_mask, input_list_word_mask)
            - sampled video frames (video, video_mask)
            - temporal range (starts, ends)
            - metadata (feature_idx, deterministic hash of video_id)
        """
        video_id, sub_id = self.iter2video_pairs_dict[feature_idx]
        input_ids, input_mask, input_list_ids, input_list_mask, input_list_word_mask, starts, ends = self._get_text(video_id, sub_id)

        video, video_mask = self._get_rawvideo(video_id, starts, ends)

        return (input_ids, input_mask,
                input_list_ids, input_list_mask, input_list_word_mask
                video, video_mask,
                feature_idx, hash(video_id))