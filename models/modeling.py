import os
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn

from .cluster import Att_Block_Patch, PCM
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import Transformer as TransformerClip
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, KL

# Distributed training ops for cross-GPU feature aggregation
allgather = AllGather.apply
allgather2 = AllGather2.apply


class ResidualLinear(nn.Module):
    """Residual linear block: x + (Linear -> ReLU -> Linear)(x)."""

    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(
            nn.Linear(d_int, d_int),
            nn.ReLU(inplace=True),
            nn.Linear(d_int, d_int),
        )

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x

class Model(nn.Module):
    """
    Core model for TSI-TVR (Temporal-Spatial Interaction for Text-Video Retrieval).

    This model extracts multi-granularity text and video features via CLIP, and
    achieves text-to-video retrieval through temporal-spatial interaction matching.

    Architecture:
        1. CLIP Encoder: extracts sentence/word/frame/patch level features for text/video;
        2. ActionFlow (PCM + Att_Block_Patch): progressive spatial clustering compression
           on video patch tokens;
        3. Temporal Interaction Branch: caption-sentence vs video-frame (global) and
           caption-word vs video-patch (local);
        4. Spatial Interaction Branch: query-sentence vs video-frame (global),
           query-word vs video-patch (local), query-sentence vs caption-sentence, and
           query-word vs caption-word;
        5. Learnable Fusion: dynamically weights granularity-level similarity matrices
           within each branch;
        6. KL Alignment Loss: aligns similarity distributions between the temporal and
           spatial branches.
    """

    def __init__(self, config):
        """
        Initialize all model modules.

        Key config attributes:
            - interaction: interaction type string.
            - agg_module: video frame aggregation mode, one of 'meanP' (mean pooling),
              'seqLSTM', or 'seqTransf'.
            - base_encoder: CLIP backbone variant, e.g., "ViT-B/32".
            - num_hidden_layers: number of Transformer layers for seqTransf agg_module.
            - max_words: maximum number of words per text sequence.
            - max_frames: maximum number of video frames.
        """
        super(Model, self).__init__()

        self.config = config

        self.interaction = config.interaction
        self.agg_module = getattr(config, 'agg_module', 'meanP')
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        assert backbone in _PT_NAME
        # Pretrained CLIP weights are searched in ./models/, then in the project root
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _PT_NAME[backbone])
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        # Derive model hyperparameters from pretrained CLIP weights automatically
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)  # e.g. 7 for ViT-B/32
        image_resolution = vision_patch_size * grid_size  # e.g. 224 for a 32px patch size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        # Initialize CLIP backbone (vision encoder + text encoder)
        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

        if torch.cuda.is_available():
            convert_weights(self.clip)

        # Cross-modal Transformer configuration for seqTransf frame aggregation
        cross_config = SimpleNamespace(**{
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 2048,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 512,
            "soft_t": 0.07,
        })
        cross_config.max_position_embeddings = context_length
        cross_config.hidden_size = transformer_width
        self.cross_config = cross_config

        # Optional temporal aggregation module for video frames
        if self.agg_module in ["seqLSTM", "seqTransf"]:
            # Frame-level positional embeddings for temporal ordering
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
            if self.agg_module == "seqTransf":
                # Multi-head Transformer for temporal modeling across frame sequences
                self.transformerClip = TransformerClip(width=transformer_width,
                                                       layers=config.num_hidden_layers,
                                                       heads=transformer_heads)
            if self.agg_module == "seqLSTM":
                # Unidirectional LSTM for temporal modeling across frame sequences
                self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size,
                                           hidden_size=cross_config.hidden_size,
                                           batch_first=True, bidirectional=False, num_layers=1)

        # Loss functions
        self.loss_fct = CrossEn(config)  # Symmetric contrastive loss (cross-entropy)
        self.loss_kl = KL(config)        # KL divergence loss for branch alignment

        self.apply(self.init_weights)  # Init new modules before loading pretrained weights
        self.clip.load_state_dict(state_dict, strict=False)

        # Re-derive embed dim for the ActionFlow / aggregation modules below
        embed_dim = state_dict["text_projection"].shape[1]
        self.max_words = config.max_words
        self.max_frames = config.max_frames
        self.max_captions = config.max_frames // 2

        # ActionFlow: three-layer progressive spatial clustering (sampling ratio 0.5 each layer)
        sr_vp = [0.5, 0.5, 0.5]
        self.v_pcm_p_1 = PCM(sample_ratio=sr_vp[0], embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_p_1 = Att_Block_Patch(dim=embed_dim, num_heads=8)
        self.v_pcm_p_2 = PCM(sample_ratio=sr_vp[1], embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_p_2 = Att_Block_Patch(dim=embed_dim, num_heads=8)
        self.v_pcm_p_3 = PCM(sample_ratio=sr_vp[2], embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_p_3 = Att_Block_Patch(dim=embed_dim, num_heads=8)

        # Learnable feature aggregation weight networks (one per granularity pair)
        self.qs_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1))
        self.qw_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1))
        self.cs_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1))
        self.cw_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1))
        self.vf_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1))
        self.vp_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1))

        # Learnable fusion weight parameters: 2 for temporal branch, 4 for spatial branch
        self.temporal_sims_w = nn.Parameter(torch.ones(2))
        self.spatial_sims_w = nn.Parameter(torch.ones(4))

        # Warm-start trick: copy CLIP pretrained params into the aggregation modules
        new_state_dict = OrderedDict()

        if self.agg_module in ["seqLSTM", "seqTransf"]:
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in state_dict.items():
                    if key == "positional_embedding":
                        new_state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if self.agg_module in ["seqTransf"] and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        if num_layer < config.num_hidden_layers:
                            new_state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        self.load_state_dict(new_state_dict, strict=False)

    def forward(self, query, query_word_mask, caption, caption_mask, caption_word_mask,
                video, video_frame_mask, idx=None, global_step=0):
        """
        Forward pass for training.

        Pipeline:
            1. Feature extraction: sentence/word-level text features for query/caption,
               and frame/patch-level visual features for video.
            2. Spatial clustering (ActionFlow): progressively compress video patch tokens
               via PCM + Att_Block_Patch across three layers.
            3. Temporal interaction: fuse caption-sentence vs video-frame (global)
               and caption-word vs video-patch (local) with learned fusion weights.
            4. Spatial interaction: fuse query-sentence vs video-frame (global),
               query-word vs video-patch (local), query-sentence vs caption-sentence,
               and query-word vs caption-word with learned fusion weights.
            5. KL alignment: minimize distribution gap between temporal and spatial similarities.
            6. Total loss: sum of temporal, spatial, and KL losses.

        Args:
            query (Tensor): query text token IDs, reshaped to [a, L].
            query_word_mask (Tensor): query text padding mask, [a, L].
            caption (list[Tensor]): caption text token ID list, length equals
                the number of caption variants (e.g., subject + whole = 2).
            caption_mask (list[Tensor]): caption sentence mask list.
            caption_word_mask (list[Tensor]): caption word mask list.
            video (Tensor): video frames, shape [b, n_v, channel, h, w] or
                [b, pair, bs, ts, channel, h, w] depending on data loader.
            video_frame_mask (Tensor): video frame mask, [b, n_v].
            idx (Tensor, optional): sample indices for contrastive learning.
            global_step (int, optional): current training step.

        Returns:
            Tensor: total training loss in training mode; None otherwise.
        """
        # Flatten local batch dimensions for uniform processing
        query = query.reshape(-1, query.shape[-1])
        query_word_mask = query_word_mask.reshape(-1, query_word_mask.shape[-1])
        caption = [c.reshape(-1, c.shape[-1]) for c in caption]
        caption_word_mask = [c.reshape(-1, c.shape[-1]) for c in caption_word_mask]

        video = torch.as_tensor(video).float()
        video_frame_mask = video_frame_mask.reshape(-1, video_frame_mask.shape[-1])

        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.reshape(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.reshape(b * pair * bs * ts, channel, h, w)

        # ========== Step 1: multi-granularity feature extraction ==========
        # qs_feat: [a, d], qw_feat: [a, w, d] — query sentence and word features
        qs_feat, qw_feat = self.get_text_feat(query, query_word_mask)

        # cs_feat: [a, c, d], cw_feat: [a, c, w, d] — caption sentence and word features
        cs_feat, cw_feat = zip(*[self.get_text_feat(c, c_mask)
                                for c, c_mask in zip(caption, caption_word_mask)])

        # vf_feat: [b, f, d], vp_feat: [b, p, d] — video frame and patch features
        vf_feat, vp_feat = self.get_video_feat(video, video_frame_mask)

        # Build granularity masks (qs_mask always valid since query is a single sentence)
        qs_mask, qw_mask = qs_feat.new_ones(qs_feat.size(0), 1), query_word_mask
        cs_mask, cw_mask = caption_mask, caption_word_mask
        vf_mask, vp_mask = video_frame_mask, vf_feat.new_ones(vf_feat.size(0), vf_feat.size(1), 7)

        # Ensure memory-contiguous layout for efficient computation
        qs_feat, qw_feat, qs_mask, qw_mask = [x.contiguous()
                            for x in [qs_feat, qw_feat, qs_mask, qw_mask]]
        cs_feat, cw_feat, cs_mask, cw_mask = [torch.stack(x, dim=1).contiguous()
                            for x in [cs_feat, cw_feat, cs_mask, cw_mask]]
        vf_feat, vp_feat, vf_mask, vp_mask = [x.contiguous()
                            for x in [vf_feat, vp_feat, vf_mask, vp_mask]]

        # Gather features across all GPUs for distributed contrastive training
        # [a, d], [a, w, d], [a, 1], [a, w]
        qs_feat, qw_feat, qs_mask, qw_mask = [allgather(x, self.config)
                            for x in [qs_feat, qw_feat, qs_mask, qw_mask]]
        # [a, c, d], [a, c, w, d], [a, c], [a, c, w]
        cs_feat, cw_feat, cs_mask, cw_mask = [allgather(x, self.config)
                            for x in [cs_feat, cw_feat, cs_mask, cw_mask]]
        
        # [b, f, d], [b, f, p, d], [b, f], [b, f, p]
        vf_feat, vp_feat, vf_mask, vp_mask = [allgather(x, self.config)
                            for x in [vf_feat, vp_feat, vf_mask, vp_mask]]
        torch.distributed.barrier()  # Synchronize all GPUs before computing losses

        # Dimension aliases for readability
        a, s, c, w = qs_feat.size(0), 1, cs_feat.size(1), qw_feat.size(1)
        b, f, p, d = vf_feat.size(0), vf_feat.size(1), vp_feat.size(1), vp_feat.size(-1)

        # CLIP-learnable temperature parameter for scaling similarity logits
        logit_scale = self.clip.logit_scale.exp()

        # ========== Step 2: ActionFlow spatial clustering (PCM) ==========
        # Merge frame dim into batch dim for per-frame patch clustering
        vp_feat = vp_feat.reshape(vf_feat.size(0) * vf_feat.size(1), -1, vp_feat.size(-1))  # [b*f, p, d]
        vp_idx_token = torch.arange(vp_feat.size(1))[None, :].repeat(vp_feat.size(0), 1)
        vp_agg_weight = vp_feat.new_ones(vp_feat.size(0), vp_feat.size(1), 1)
        vp_mask = vp_feat.new_ones(vp_feat.size(0), vp_feat.size(1))
        vp_token_dict = {
            'x': vp_feat,
            'token_num': vp_feat.size(1),
            'idx_token': vp_idx_token,
            'agg_weight': vp_agg_weight,
            'mask': vp_mask.detach()
        }
        # Three-layer progressive PCM clustering with attention refinement
        vp_token_dict = self.v_att_block_p_1(self.v_pcm_p_1(vp_token_dict))
        vp_token_dict = self.v_att_block_p_2(self.v_pcm_p_2(vp_token_dict))
        vp_token_dict = self.v_att_block_p_3(self.v_pcm_p_3(vp_token_dict))
        vp_feat = vp_token_dict['x']
        # Restore frame dimension: [b, f, p', d] where p' < p after clustering
        vp_feat = vp_feat.reshape(vf_feat.size(0), vf_feat.size(1), -1, vp_feat.size(-1))
        vp_mask = vp_feat.new_ones(vf_feat.size(0), vf_feat.size(1), vp_feat.size(2))  # [b, f, p']

        # (Optional) order augmentation: shuffle frames / captions (disabled by default)
        # vf_feat, vf_mask, vp_feat, vp_mask = self.shuffle_frames(vf_feat, vf_mask, vp_feat, vp_mask)
        # cs_feat, cs_mask, cw_feat, cw_mask = self.shuffle_captions(cs_feat, cs_mask, cw_feat, cw_mask)

        # ========== Step 3: temporal interaction branch (caption vs video) ==========
        sims_cs_vf = self.cs_and_vf(cs_feat, cs_mask, vf_feat, vf_mask)  # [a, b]
        sims_cw_vp = self.cw_and_vp(cw_feat, cw_mask, vp_feat, vp_mask)  # [a, b]

        # Softmax-normalized learnable fusion weights for temporal branch
        temporal_sims_w = torch.softmax(self.temporal_sims_w, dim=0)
        sims_temporal = (temporal_sims_w[0] * sims_cs_vf +
                         temporal_sims_w[1] * sims_cw_vp)

        # Symmetric contrastive loss for temporal branch with temperature scaling
        loss_temporal = (self.loss_fct(sims_temporal * logit_scale) +
                         self.loss_fct(sims_temporal.T * logit_scale)) / 2.0

        # ========== Step 4: spatial interaction branch (query vs caption + query vs video) ==========
        sims_qs_cs = self.qs_and_cs(qs_feat, qs_mask, cs_feat, cs_mask)  # [a, a]
        sims_qw_cw = self.qw_and_cw(qw_feat, qw_mask, cw_feat, cw_mask)  # [a, a]
        sims_qs_vf = self.qs_and_vf(qs_feat, qs_mask, vf_feat, vf_mask)  # [a, b]
        sims_qw_vp = self.qw_and_vp(qw_feat, qw_mask, vp_feat, vp_mask)  # [a, b]

        # Softmax-normalized learnable fusion weights for spatial branch
        spatial_sims_w = torch.softmax(self.spatial_sims_w, dim=0)
        sims_spatial = (spatial_sims_w[0] * sims_qs_cs +
                        spatial_sims_w[1] * sims_qw_cw +
                        spatial_sims_w[2] * sims_qs_vf +
                        spatial_sims_w[3] * sims_qw_vp)

        # Symmetric contrastive loss for spatial branch with temperature scaling
        loss_spatial = (self.loss_fct(sims_spatial * logit_scale) +
                        self.loss_fct(sims_spatial.T * logit_scale)) / 2.0

        # ========== Step 5: total loss with KL alignment ==========
        # KL divergence between temporal and spatial similarity distributions (bidirectional)
        loss_kl = (self.loss_kl(sims_temporal, sims_spatial) +
                   self.loss_kl(sims_temporal.T, sims_spatial.T) + 
                   self.loss_kl(sims_spatial, sims_temporal) + 
                   self.loss_kl(sims_spatial.T, sims_temporal.T)) / 4.0

        total_loss = loss_temporal + loss_spatial + loss_kl

        if self.training:
            return total_loss
        else:
            return None

    def get_text_feat(self, text_ids, text_mask):
        """
        Extract multi-granularity text features through CLIP.

        Args:
            text_ids (Tensor): text token IDs, shape [bs_pair, L].
            text_mask (Tensor): text attention mask, shape [bs_pair, L].

        Returns:
            tuple:
                - s_feat (Tensor): sentence-level [CLS] features, [bs_pair, d].
                - w_feat (Tensor): word-level hidden features, [bs_pair, num_words, d].
        """
        text_ids = text_ids.reshape(-1, text_ids.shape[-1])
        text_mask = text_mask.reshape(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)
        s_feat, w_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)
        s_feat = s_feat.float().reshape(bs_pair, s_feat.size(-1))
        w_feat = w_feat.float().reshape(bs_pair, -1, w_feat.size(-1))
        return s_feat, w_feat

    def get_video_feat(self, video, video_mask):
        """
        Extract multi-granularity video features through CLIP.

        Args:
            video (Tensor): video frame pixels, shape [bs_pair * n_v, channel, h, w].
            video_mask (Tensor): video frame mask, shape [bs_pair, n_v].

        Returns:
            tuple:
                - f_feat (Tensor): frame-level [CLS] features, [bs_pair, num_frames, d].
                - p_feat (Tensor): patch-level hidden features, [bs_pair, num_patches, d].
        """
        if not self.training:
            # Reshape inputs for inference mode to match training dimensions
            video_mask = video_mask.reshape(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.size()) == 5:
                b, n_v, d, h, w = video.shape
                video = video.reshape(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.reshape(b * pair * bs * ts, channel, h, w)

        bs_pair, n_v = video_mask.size()
        f_feat, p_feat = self.clip.encode_image(video, return_hidden=True, mask=video_mask)
        f_feat = f_feat.float().reshape(bs_pair, -1, f_feat.size(-1))
        f_feat = self.agg_video_feat(f_feat, video_mask, self.agg_module)
        p_feat = p_feat.float().reshape(bs_pair, -1, p_feat.size(-1))
        return f_feat, p_feat
    
    def agg_video_feat(self, video_feat, video_mask, agg_module):
        """Aggregate frame-level features into a unified temporal representation.

        Supports three aggregation strategies:
            - "None"   : identity (no aggregation).
            - "seqLSTM": unidirectional LSTM with residual connection.
            - "seqTransf": multi-head Transformer encoder with positional
              embeddings and residual connection.

        Args:
            video_feat (Tensor): frame features, shape [bs, num_frames, d].
            video_mask (Tensor): frame validity mask, shape [bs, num_frames].
            agg_module (str): aggregation mode, one of "None", "seqLSTM",
                "seqTransf".

        Returns:
            Tensor: aggregated frame features, same shape as input [bs, num_frames, d].
        """
        video_feat = video_feat.contiguous()
        if agg_module == "None":
            pass
        elif agg_module == "seqLSTM":
            # Sequential type: LSTM
            video_feat_original = video_feat
            video_feat = pack_padded_sequence(video_feat, torch.sum(video_mask, dim=-1).cpu(),
                                              batch_first=True, enforce_sorted=False)
            video_feat, _ = self.lstm_visual(video_feat)
            if self.training: self.lstm_visual.flatten_parameters()
            video_feat, _ = pad_packed_sequence(video_feat, batch_first=True)
            video_feat = torch.cat(
                (video_feat, video_feat_original[:, video_feat.size(1):, ...].contiguous()), dim=1)
            video_feat = video_feat + video_feat_original
        elif agg_module == "seqTransf":
            # Sequential type: Transformer Encoder
            video_feat_original = video_feat
            seq_length = video_feat.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=video_feat.device)
            position_ids = position_ids.unsqueeze(0).expand(video_feat.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            video_feat = video_feat + frame_position_embeddings
            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            video_feat = video_feat.permute(1, 0, 2)  # NLD -> LND
            video_feat = self.transformerClip(video_feat, extended_video_mask)
            video_feat = video_feat.permute(1, 0, 2)  # LND -> NLD
            video_feat = video_feat + video_feat_original
        return video_feat

    def norm(self, feat):
        """Apply L2 normalization along the last (feature) dimension.

        Maps each feature vector to the unit hypersphere so that dot-product
        similarity equals cosine similarity.

        Args:
            feat (Tensor): input features, arbitrary shape [..., d].

        Returns:
            Tensor: L2-normalized features, same shape as input.
        """
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat
    
    def shuffle_frames(self, vf_feat, vf_mask, vp_feat, vp_mask):
        """Randomly permute the temporal (frame) dimension across all inputs.

        Applies the same random permutation to frame features, frame masks,
        patch features, and patch masks so that temporal alignment is preserved.

        Args:
            vf_feat (Tensor): frame features, shape [b, f, d].
            vf_mask (Tensor): frame mask, shape [b, f].
            vp_feat (Tensor): patch features, shape [b, f, p, d].
            vp_mask (Tensor): patch mask, shape [b, f, p].

        Returns:
            tuple: permuted (vf_feat, vf_mask, vp_feat, vp_mask).
        """
        f = vf_feat.size(1)
        perm = torch.randperm(f, device=vf_feat.device)

        vf_feat = vf_feat[:, perm]
        vf_mask = vf_mask[:, perm]
        vp_feat = vp_feat[:, perm]
        vp_mask = vp_mask[:, perm]

        return vf_feat, vf_mask, vp_feat, vp_mask

    def shuffle_captions(self, cs_feat, cs_mask, cw_feat, cw_mask):
        """Randomly permute the caption dimension across all inputs.

        Applies the same random permutation to caption sentence features,
        caption masks, word features, and word masks so that caption-level
        alignment is preserved.

        Args:
            cs_feat (Tensor): caption sentence features, shape [a, c, d].
            cs_mask (Tensor): caption sentence mask, shape [a, c].
            cw_feat (Tensor): caption word features, shape [a, c, w, d].
            cw_mask (Tensor): caption word mask, shape [a, c, w].

        Returns:
            tuple: permuted (cs_feat, cs_mask, cw_feat, cw_mask).
        """
        c = cs_feat.size(1)
        perm = torch.randperm(c, device=cs_feat.device)

        cs_feat = cs_feat[:, perm]
        cs_mask = cs_mask[:, perm]
        cw_feat = cw_feat[:, perm]
        cw_mask = cw_mask[:, perm]

        return cs_feat, cs_mask, cw_feat, cw_mask

    def cs_and_vf(self, cs_feat, cs_mask, vf_feat, vf_mask):
        """
        Global-level temporal interaction: caption-sentence <-> video-frame.

        Symmetric attention mechanism:
            1) cs -> vf: best-matching frame per caption, then caption-weighted aggregate;
            2) vf -> cs: best-matching caption per frame, then frame-weighted aggregate.
        Final similarity is the average of both directions.

        Args:
            cs_feat (Tensor): caption sentence features, shape [a, c, d].
            cs_mask (Tensor): caption sentence mask, shape [a, c].
            vf_feat (Tensor): video frame features, shape [b, f, d].
            vf_mask (Tensor): video frame mask, shape [b, f].

        Returns:
            Tensor: similarity matrix, shape [a, b].
        """
        # Compute learnable aggregation weights for each caption sentence
        cs_feat_w = self.cs_feat_w(cs_feat).squeeze(-1)  # [a, c]
        cs_feat_w.masked_fill_((1 - cs_mask).to(torch.bool), float(-9e15))
        cs_feat_w = torch.softmax(cs_feat_w, dim=-1)

        # Compute learnable aggregation weights for each video frame
        vf_feat_w = self.vf_feat_w(vf_feat).squeeze(-1)  # [b, f]
        vf_feat_w.masked_fill_((1 - vf_mask).to(torch.bool), float(-9e15))
        vf_feat_w = torch.softmax(vf_feat_w, dim=-1)

        # Compute full pairwise similarity tensor [a, b, c, f]
        sims_cs_vf = torch.einsum("acd,bfd->abcf", [self.norm(cs_feat), self.norm(vf_feat)])
        sims_cs_vf = torch.einsum('abcf,ac->abcf', [sims_cs_vf, cs_mask])
        sims_cs_vf = torch.einsum('abcf,bf->abcf', [sims_cs_vf, vf_mask])

        # Direction: caption -> video, max over frame dim then caption-weighted
        sims_cs2vf, _ = sims_cs_vf.max(dim=-1)  # [a, b, c]
        sims_cs2vf = torch.einsum('abc,ac->ab', [sims_cs2vf, cs_feat_w])

        # Direction: video -> caption, max over caption dim then frame-weighted
        sims_vf2cs, _ = sims_cs_vf.max(dim=-2)  # [a, b, f]
        sims_vf2cs = torch.einsum('abf,bf->ab', [sims_vf2cs, vf_feat_w])

        sims_cs_vf = (sims_cs2vf + sims_vf2cs) / 2.0
        return sims_cs_vf

    def cw_and_vp(self, cw_feat, cw_mask, vp_feat, vp_mask):
        """
        Local-level temporal interaction: caption-word <-> video-patch.

        Symmetric attention mechanism:
            1) cw -> vp: best-matching patch then frame, then word-weighted aggregate;
            2) vp -> cw: best-matching word then caption, then patch-weighted aggregate.
        Final similarity is the average of both directions.

        Args:
            cw_feat (Tensor): caption word features, shape [a, c, w, d].
            cw_mask (Tensor): caption word mask, shape [a, c, w].
            vp_feat (Tensor): video patch features, shape [b, f, p, d].
            vp_mask (Tensor): video patch mask, shape [b, f, p].

        Returns:
            Tensor: similarity matrix, shape [a, b].
        """
        # Compute learnable aggregation weights for each caption word
        cw_feat_w = self.cw_feat_w(cw_feat).squeeze(-1)  # [a, c, w]
        cw_feat_w.masked_fill_((1 - cw_mask).to(torch.bool), float(-9e15))
        cw_feat_w = torch.softmax(cw_feat_w, dim=-1)

        # Compute learnable aggregation weights for each video patch
        vp_feat_w = self.vp_feat_w(vp_feat).squeeze(-1)  # [b, f, p]
        vp_feat_w.masked_fill_((1 - vp_mask).to(torch.bool), float(-9e15))
        vp_feat_w = torch.softmax(vp_feat_w, dim=-1)

        # Compute full pairwise similarity tensor [a, b, c, w, f, p]
        sims_cw_vp = torch.einsum("acwd,bfpd->abcwfp", [self.norm(cw_feat), self.norm(vp_feat)])
        sims_cw_vp = torch.einsum('abcwfp,acw->abcwfp', [sims_cw_vp, cw_mask])
        sims_cw_vp = torch.einsum('abcwfp,bfp->abcwfp', [sims_cw_vp, vp_mask])

        # Direction: caption-word -> video-patch, max over patch then frame, then word-weighted
        sims_cw2vp, _ = sims_cw_vp.max(dim=-1)  # [a, b, c, w, f]
        sims_cw2vp, _ = sims_cw2vp.max(dim=-1)  # [a, b, c, w]
        sims_cw2vp = torch.einsum('abcw,acw->ab', [sims_cw2vp, cw_feat_w])

        # Direction: video-patch -> caption-word, max over word then caption, then patch-weighted
        sims_vp2cw, _ = sims_cw_vp.max(dim=-3)  # [a, b, c, f, p]
        sims_vp2cw, _ = sims_vp2cw.max(dim=-3)  # [a, b, f, p]
        sims_vp2cw = torch.einsum('abfp,bfp->ab', [sims_vp2cw, vp_feat_w])

        sims_cw_vp = (sims_cw2vp + sims_vp2cw) / 2.0
        return sims_cw_vp

    def qs_and_vf(self, qs_feat, qs_mask, vf_feat, vf_mask):
        """
        Global-level spatial interaction: query-sentence <-> video-frame.

        Symmetric attention mechanism:
            1) qs -> vf: best-matching frame per query via max aggregation;
            2) vf -> qs: weighted frame aggregation.
        Final similarity is the average of both directions.

        Args:
            qs_feat (Tensor): query sentence features, shape [a, d].
            qs_mask (Tensor): query sentence mask, shape [a, 1].
            vf_feat (Tensor): video frame features, shape [b, f, d].
            vf_mask (Tensor): video frame mask, shape [b, f].

        Returns:
            Tensor: similarity matrix, shape [a, b].
        """
        # Compute learnable aggregation weights for each video frame
        vf_feat_w = self.vf_feat_w(vf_feat).squeeze(-1)  # [b, f]
        vf_feat_w.masked_fill_((1 - vf_mask).to(torch.bool), float(-9e15))
        vf_feat_w = torch.softmax(vf_feat_w, dim=-1)

        # Compute pairwise similarity [a, b, f]
        sims_qs_vf = torch.einsum("ad,bfd->abf", [self.norm(qs_feat), self.norm(vf_feat)])
        sims_qs_vf = torch.einsum('abf,bf->abf', [sims_qs_vf, vf_mask])

        # Direction: query -> video, max over frame dimension
        sims_qs2vf, _ = sims_qs_vf.max(dim=-1)  # [a, b]

        # Direction: video -> query, frame-weighted aggregation
        sims_vf2qs = torch.einsum('abf,bf->ab', [sims_qs_vf, vf_feat_w])  # [a, b]

        sims_qs_vf = (sims_qs2vf + sims_vf2qs) / 2.0
        return sims_qs_vf

    def qw_and_vp(self, qw_feat, qw_mask, vp_feat, vp_mask):
        """
        Local-level spatial interaction: query-word <-> video-patch.

        Symmetric attention mechanism:
            1) qw -> vp: best-matching patch then frame, then word-weighted aggregate;
            2) vp -> qw: best-matching word, then patch-weighted aggregate.
        Final similarity is the average of both directions.

        Args:
            qw_feat (Tensor): query word features, shape [a, w, d].
            qw_mask (Tensor): query word mask, shape [a, w].
            vp_feat (Tensor): video patch features, shape [b, f, p, d].
            vp_mask (Tensor): video patch mask, shape [b, f, p].

        Returns:
            Tensor: similarity matrix, shape [a, b].
        """
        # Compute learnable aggregation weights for each query word
        qw_feat_w = self.qw_feat_w(qw_feat).squeeze(-1)  # [a, w]
        qw_feat_w.masked_fill_((1 - qw_mask).to(torch.bool), float(-9e15))
        qw_feat_w = torch.softmax(qw_feat_w, dim=-1)

        # Compute learnable aggregation weights for each video patch
        vp_feat_w = self.vp_feat_w(vp_feat).squeeze(-1)  # [b, f, p]
        vp_feat_w.masked_fill_((1 - vp_mask).to(torch.bool), float(-9e15))
        vp_feat_w = torch.softmax(vp_feat_w, dim=-1)

        # Compute pairwise similarity [a, b, w, f, p]
        sims_qw_vp = torch.einsum("awd,bfpd->abwfp", [self.norm(qw_feat), self.norm(vp_feat)])
        sims_qw_vp = torch.einsum('abwfp,aw->abwfp', [sims_qw_vp, qw_mask])
        sims_qw_vp = torch.einsum('abwfp,bfp->abwfp', [sims_qw_vp, vp_mask])

        # Direction: query-word -> video-patch, max over patch then frame, then word-weighted
        sims_qw2vp, _ = sims_qw_vp.max(dim=-1)  # [a, b, w, f]
        sims_qw2vp, _ = sims_qw2vp.max(dim=-1)  # [a, b, w]
        sims_qw2vp = torch.einsum('abw,aw->ab', [sims_qw2vp, qw_feat_w])

        # Direction: video-patch -> query-word, max over word then patch-weighted
        sims_vp2qw, _ = sims_qw_vp.max(dim=-3)  # [a, b, f, p]
        sims_vp2qw = torch.einsum('abfp,bfp->ab', [sims_vp2qw, vp_feat_w])

        sims_qw_vp = (sims_qw2vp + sims_vp2qw) / 2.0
        return sims_qw_vp

    def qs_and_cs(self, qs_feat, qs_mask, cs_feat, cs_mask):
        """
        Global-level text-text interaction: query-sentence <-> caption-sentence.

        Symmetric attention mechanism:
            1) qs -> cs: best-matching caption per query via max aggregation;
            2) cs -> qs: weighted caption aggregation.
        Final similarity is the average of both directions.

        Args:
            qs_feat (Tensor): query sentence features, shape [a, d].
            qs_mask (Tensor): query sentence mask, shape [a, 1].
            cs_feat (Tensor): caption sentence features, shape [a, c, d].
            cs_mask (Tensor): caption sentence mask, shape [a, c].

        Returns:
            Tensor: similarity matrix, shape [a, a].
        """
        # Compute learnable aggregation weights for each caption sentence
        cs_feat_w = self.cs_feat_w(cs_feat).squeeze(-1)  # [a, c]
        cs_feat_w.masked_fill_((1 - cs_mask).to(torch.bool), float(-9e15))
        cs_feat_w = torch.softmax(cs_feat_w, dim=-1)

        # Compute pairwise similarity [a, a, c]
        sims = torch.einsum('ed,jcd->ejc', [self.norm(qs_feat), self.norm(cs_feat)])
        sims = torch.einsum('ejc,jc->ejc', [sims, cs_mask])

        # Direction: query -> caption, max over caption dimension
        sims_qs2cs, _ = sims.max(dim=-1)  # [a, a]

        # Direction: caption -> query, weighted over caption dimension
        sims_cs2qs = torch.einsum('ejc,jc->ej', [sims, cs_feat_w])  # [a, a]

        return (sims_qs2cs + sims_cs2qs) / 2.0

    def qw_and_cw(self, qw_feat, qw_mask, cw_feat, cw_mask):
        """
        Local-level text-text interaction: query-word <-> caption-word.

        Symmetric attention mechanism:
            1) qw -> cw: best-matching word then caption, then query-word-weighted;
            2) cw -> qw: best-matching query-word, then caption-word-weighted.
        Final similarity is the average of both directions.

        Args:
            qw_feat (Tensor): query word features, shape [a, w, d].
            qw_mask (Tensor): query word mask, shape [a, w].
            cw_feat (Tensor): caption word features, shape [a, c, w, d].
            cw_mask (Tensor): caption word mask, shape [a, c, w].

        Returns:
            Tensor: similarity matrix, shape [a, a].
        """
        # Compute learnable aggregation weights for each query word
        qw_feat_w = self.qw_feat_w(qw_feat).squeeze(-1)  # [a, w]
        qw_feat_w.masked_fill_((1 - qw_mask).to(torch.bool), float(-9e15))
        qw_feat_w = torch.softmax(qw_feat_w, dim=-1)

        # Compute learnable aggregation weights for each caption word
        cw_feat_w = self.cw_feat_w(cw_feat).squeeze(-1)  # [a, c, w]
        cw_feat_w.masked_fill_((1 - cw_mask).to(torch.bool), float(-9e15))
        cw_feat_w = torch.softmax(cw_feat_w, dim=-1)

        # Compute pairwise similarity [a, a, c, w_q, w_c]
        sims = torch.einsum("eid,jckd->ejcik", [self.norm(qw_feat), self.norm(cw_feat)])
        sims = torch.einsum('ejcik,ei->ejcik', [sims, qw_mask])
        sims = torch.einsum('ejcik,jck->ejcik', [sims, cw_mask])

        # Direction: query-word -> caption-word, max word then caption, then qw-weighted
        sims_qw2cw, _ = sims.max(dim=-1)  # [a, a, c, w_q]
        sims_qw2cw, _ = sims_qw2cw.max(dim=-2)  # [a, a, w_q]
        sims_qw2cw = torch.einsum('eji,ei->ej', [sims_qw2cw, qw_feat_w])  # [a, a]

        # Direction: caption-word -> query-word, max over qw then cw-weighted
        sims_cw2qw, _ = sims.max(dim=-2)  # [a, a, c, w_c]
        sims_cw2qw = torch.einsum('ejck,jck->ej', [sims_cw2qw, cw_feat_w])  # [a, a]

        return (sims_qw2cw + sims_cw2qw) / 2.0

    def get_similarity_logits(self, qs_feat, qw_feat, qw_mask,
                              cs_feat, cs_mask, cw_feat, cw_mask,
                              vf_feat, vf_mask, vp_feat):
        """
        Compute final similarity logits for inference.

        Mirrors the spatial interaction branch of forward():
            1) PCM spatial clustering on video patch tokens;
            2) qs-cs global text-text matching;
            3) qw-cw local text-text matching;
            4) qs-vf global text-video matching;
            5) qw-vp local text-video matching;
            6) Fuse all four similarity matrices with learned weights.

        Args:
            qs_feat (Tensor): query sentence features, [a, d].
            qw_feat (Tensor): query word features, [a, w, d].
            qw_mask (Tensor): query word mask, [a, w].
            cs_feat (Tensor): caption sentence features, [a, c, d].
            cs_mask (Tensor): caption sentence mask, [a, c].
            cw_feat (Tensor): caption word features, [a, c, w, d].
            cw_mask (Tensor): caption word mask, [a, c, w].
            vf_feat (Tensor): video frame features, [b, f, d].
            vf_mask (Tensor): video frame mask, [b, f].
            vp_feat (Tensor): video patch features (pre-PCM), [b, f, p, d].

        Returns:
            Tensor: similarity matrix, shape [a, b].
        """
        # qs_mask is always valid since query has a single sentence
        a = qs_feat.size(0)
        b, f, p, d = vf_feat.size(0), vf_feat.size(1), vp_feat.size(2), vp_feat.size(-1)
        qs_mask = qs_feat.new_ones(a, 1)

        # ActionFlow: three-layer PCM clustering identical to training (forward)
        vp_feat = vp_feat.reshape(b * f, -1, d)  # [b*f, p, d]
        vp_idx_token = torch.arange(vp_feat.size(1), device=vp_feat.device)[None, :].repeat(vp_feat.size(0), 1)
        vp_agg_weight = vp_feat.new_ones(vp_feat.size(0), vp_feat.size(1), 1)
        vp_mask_pcm = vp_feat.new_ones(vp_feat.size(0), vp_feat.size(1))
        vp_token_dict = {
            'x': vp_feat,
            'token_num': vp_feat.size(1),
            'idx_token': vp_idx_token,
            'agg_weight': vp_agg_weight,
            'mask': vp_mask_pcm.detach()
        }
        vp_token_dict = self.v_att_block_p_1(self.v_pcm_p_1(vp_token_dict))
        vp_token_dict = self.v_att_block_p_2(self.v_pcm_p_2(vp_token_dict))
        vp_token_dict = self.v_att_block_p_3(self.v_pcm_p_3(vp_token_dict))
        vp_feat = vp_token_dict['x']
        vp_feat = vp_feat.reshape(b, f, -1, d)  # [b, f, p', d]
        vp_mask = vf_feat.new_ones(b, f, vp_feat.size(2))  # [b, f, p']

        # Compute all four spatial interaction similarity matrices
        sims_qs_cs = self.qs_and_cs(qs_feat, qs_mask, cs_feat, cs_mask)  # [a, a]
        sims_qw_cw = self.qw_and_cw(qw_feat, qw_mask, cw_feat, cw_mask)  # [a, a]
        sims_qs_vf = self.qs_and_vf(qs_feat, qs_mask, vf_feat, vf_mask)  # [a, b]
        sims_qw_vp = self.qw_and_vp(qw_feat, qw_mask, vp_feat, vp_mask)  # [a, b]

        # Fuse with softmax-normalized learnable weights
        spatial_sims_w = torch.softmax(self.spatial_sims_w, dim=0)
        sims_spatial = (spatial_sims_w[0] * sims_qs_cs +
                        spatial_sims_w[1] * sims_qw_cw +
                        spatial_sims_w[2] * sims_qs_vf +
                        spatial_sims_w[3] * sims_qw_vp)
        return sims_spatial

    @property
    def dtype(self):
        """Return the dtype of the first model parameter.

        Falls back to scanning all tensor attributes if no parameters exist
        (e.g., for an empty module).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """Initialize weights for newly added modules.

        Applies:
            - Normal init (mean=0, std=0.02) to Linear and Embedding weights;
            - Zero bias for Linear layers;
            - Zero beta / one gamma (or zero bias / one weight) for LayerNorm.

        Intended to be passed to ``self.apply(self.init_weights)`` before
        loading pretrained CLIP weights so that only non-pretrained parameters
        are randomly initialized.

        Args:
            module (nn.Module): submodule to initialize.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
