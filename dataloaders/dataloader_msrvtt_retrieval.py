"""MSR-VTT dataset loader for video-text retrieval.

MSR-VTT (Microsoft Research Video to Text) is a standard benchmark for
video-to-text retrieval. It contains 10K web video clips paired with
~200K natural-language captions. Each video has multiple annotations
covering visual content, actions, and temporal events.

Two levels of textual granularity are provided:
  1. Caption-level — a single sentence that describes the entire video.
  2. Sentence-level temporal descriptions (seq_cap) — up to 10 short
     phrases produced by VILA / Chat-UniVi-13B, each corresponding to a
     distinct temporal segment. Shorter lists are padded to a fixed length
     by repeating the last fragment; a binary mask distinguishes real
     entries from padding.

Training split (9K videos):
  Primary captions are loaded from MSRVTT_data.json. Each training sample
  consists of (query_sentence, video_id, temporal_sentences, sentence_mask).

Test split (1K videos):
  Primary captions come from MSRVTT_test.1000.csv. Each test sample
  contains the ground-truth caption and the corresponding temporal
  sentences.

Expected directory layout:
    anno_path/
        MSRVTT_train.9000.csv               # training video ID list
        MSRVTT_test.1000.csv                # test video ID list
        MSRVTT_data.json                    # training annotations (200K sentences)
        captions/MSRVTT_Chat-UniVi-13B.json # temporal sentence descriptions
    video_path/
        {video_id}.mp4                      # video files
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import pandas as pd
from os.path import join, exists
from collections import OrderedDict
from .dataloader_retrieval import RetrievalDataset


class MSRVTTDataset(RetrievalDataset):
    """MSR-VTT dataset loader for text-video retrieval.

    Extends RetrievalDataset by overriding ``_get_anns`` to parse MSR-VTT's
    annotation files and build a pre-indexed dictionary for O(1) temporal
    caption lookup.
    """

    def __init__(self, subset, anno_path, video_path, tokenizer, max_words=32,
                 max_frames=12, video_framerate=1, image_resolution=224, mode='all', config=None):
        """Initialize the MSR-VTT dataset.

        Args:
            subset (str): Dataset split — 'train' (9K videos) or 'test' (1K videos).
            anno_path (str): Root directory containing all annotation files.
            video_path (str): Directory containing video MP4 files.
            tokenizer: Text tokenizer for converting sentences to token IDs.
            max_words (int): Maximum number of tokens per text input.
            max_frames (int): Maximum number of video frames to sample.
            video_framerate (int): Frame sampling rate in frames per second.
            image_resolution (int): Spatial resolution of resized video frames.
            mode (str): Sampling mode passed to the parent class.
            config: Optional configuration object.
        """
        super(MSRVTTDataset, self).__init__(subset, anno_path, video_path, tokenizer,
                 max_words, max_frames, video_framerate, image_resolution, mode, config=config)

    def _get_anns(self, subset='train'):
        """Load video paths and caption annotations for the given subset.

        Returns two OrderedDicts consumed by the parent dataloader:
            video_dict     : {video_id  -> absolute path to .mp4 file}
            sentences_dict : {sample_idx -> (video_id, caption_tuple)}
                             caption_tuple is a 5-tuple:
                               (caption_text, None, None, seq_cap_list, seq_cap_mask)
                             - caption_text : primary sentence describing the video.
                             - seq_cap_list : list of up to 10 temporal sentence fragments.
                             - seq_cap_mask : binary list; 1 = real, 0 = padding.

        The temporal descriptions are pre-indexed into a dict keyed by video_id
        for O(1) lookup during annotation matching.

        Args:
            subset (str): 'train' (9K videos) or 'test' (1K videos).

        Returns:
            tuple: (video_dict, sentences_dict).

        Raises:
            FileNotFoundError: If the required CSV file does not exist.
        """
        # Step 1: Locate and read the video ID list for the requested split.
        csv_path = {
            'train': join(self.anno_path, 'MSRVTT_train.9000.csv'),
            'test':  join(self.anno_path, 'MSRVTT_test.1000.csv'),
        }[subset]
        if not exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        csv = pd.read_csv(csv_path)
        video_id_list = list(csv['video_id'].values)

        video_dict = OrderedDict()
        sentences_dict = OrderedDict()

        # Step 2: Load temporal sentence-level descriptions from VILA /
        # Chat-UniVi-13B and build a hash index for O(1) video_id lookup.
        #   - Each video's description string is split on '.' into fragments.
        #   - Shorter lists are padded to `caps_len` by repeating the last fragment.
        #   - A binary mask tracks which entries are real vs. padded.
        caps_json_path = 'captions/MSRVTT_Chat-UniVi-13B.json'
        caps_len = 10  # target length after padding
        caps_anno_path = join(self.anno_path, caps_json_path)
        with open(caps_anno_path, 'r') as f:
            caps_data = json.load(f)
        caps_lookup = {entry['name']: entry for entry in caps_data}

        # Step 3: Build per-sample annotations, branching on split.
        if subset == 'train':
            # ----------------------------------------------------------
            # Training split
            #   Primary captions come from MSRVTT_data.json. Each entry
            #   under 'sentences' is a (video_id, caption) pair. We match
            #   it with the temporal description via caps_lookup.
            # ----------------------------------------------------------
            anno_path = join(self.anno_path, 'MSRVTT_data.json')
            with open(anno_path, 'r') as f:
                data = json.load(f)

            for itm in data['sentences']:
                # Skip videos not in the current split
                if itm['video_id'] not in video_id_list:
                    continue
                # Retrieve temporal description via pre-built dict index
                cap_entry = caps_lookup.get(itm['video_id'])
                if cap_entry is None:
                    continue
                # Split description string on '.', strip whitespace, remove blanks
                cap = [s.strip() for s in cap_entry['description'].split(".") if s.strip()]
                num_real = len(cap)
                # Pad to fixed length by repeating the last fragment
                if num_real < caps_len:
                    cap.extend([cap[-1]] * (caps_len - num_real))
                cap = cap[:caps_len]
                # Binary mask: 1 for genuine sentences, 0 for padded
                cap_mask = [1] * num_real + [0] * (caps_len - num_real)
                cap_mask = cap_mask[:caps_len]
                # Store the sample: (video_id, (caption, None, None, cap, cap_mask))
                sentences_dict[len(sentences_dict)] = (itm['video_id'], (itm['caption'], None, None, cap, cap_mask))
                video_dict[itm['video_id']] = join(self.video_path, f"{itm['video_id']}.mp4")
        else:
            # ----------------------------------------------------------
            # Test split
            #   Primary captions come from the CSV column 'sentence'
            #   rather than from MSRVTT_data.json. Temporal descriptions
            #   are still matched via caps_lookup.
            # ----------------------------------------------------------
            for _, itm in csv.iterrows():
                cap_entry = caps_lookup.get(itm['video_id'])
                if cap_entry is None:
                    continue
                # Split description string on '.', strip whitespace, remove blanks
                cap = [s.strip() for s in cap_entry['description'].split(".") if s.strip()]
                num_real = len(cap)
                # Pad to fixed length by repeating the last fragment
                if num_real < caps_len:
                    cap.extend([cap[-1]] * (caps_len - num_real))
                cap = cap[:caps_len]
                # Binary mask: 1 for genuine sentences, 0 for padded
                cap_mask = [1] * num_real + [0] * (caps_len - num_real)
                cap_mask = cap_mask[:caps_len]
                # Store the sample
                sentences_dict[len(sentences_dict)] = (itm['video_id'], (itm['sentence'], None, None, cap, cap_mask))
                video_dict[itm['video_id']] = join(self.video_path, f"{itm['video_id']}.mp4")

        return video_dict, sentences_dict