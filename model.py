import math
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        # a tiny param so we always have device/dtype
        self._dummy_variable = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B] (timesteps)
        device = x.device
        half = self.dim // 2
        scale = math.log(10000) / max(1, half - 1)
        freq = torch.exp(torch.arange(half, device=device) * -scale)
        args = x[:, None].float() * freq[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def _build_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    # Build fixed 2D sin/cos positional encodings: [grid_h*grid_w, embed_dim]
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be multiple of 4 for 2D sincos, got {embed_dim}")
    half = embed_dim // 2
    emb_h = SinusoidalPosEmb(half)(torch.arange(grid_h))  # [H, half]
    emb_w = SinusoidalPosEmb(half)(torch.arange(grid_w))  # [W, half]
    emb_h = emb_h.unsqueeze(1).repeat(1, grid_w, 1)
    emb_w = emb_w.unsqueeze(0).repeat(grid_h, 1, 1)
    emb = torch.cat([emb_h, emb_w], dim=-1).view(grid_h * grid_w, embed_dim)
    return emb


class PatchEmbed(nn.Module):
    """Patchify an image with Conv2d and project to tokens."""

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # GroupNorm tends to be stable for small channel counts
        gn_groups = max(1, embed_dim // 8)
        self.norm = nn.GroupNorm(gn_groups, embed_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, P, C]
        return x, (H, W)


class DiffusionWaypointTransformer(ModuleAttrMixin):
    """
    Transformer model for diffusion noise prediction over K waypoints, conditioned on a
    traversability image. Supports encoder-only, encoder-decoder, and decoder-only modes.

    Inputs:
      - image: [B, 1, H, W]
      - sample: [B, K, 3]
      - timestep: scalar, [B] or [B,1]
    Output:
      - [B, K, 3]
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: int = 8,
        in_chans: int = 1,
        num_waypoints: int = 5,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        use_learned_img_pos: bool = False,
        # modes/conditioning
        mode: str = 'encoder_decoder',  # 'encoder', 'decoder', 'encoder_decoder'
        causal_attn: bool = False,
        time_as_cond: bool = True,
        obs_as_cond: bool = False,
        cond_dim: int = 0,
        n_cond_layers: int = 0,
        vision_n_layers: int = 0,
    ) -> None:
        super().__init__()

        if mode not in ('encoder', 'decoder', 'encoder_decoder'):
            raise ValueError(f"Unknown mode {mode}")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_waypoints = num_waypoints
        self.embed_dim = embed_dim
        self.mode = mode
        self.causal_attn = causal_attn
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond and (cond_dim > 0)
        self.cond_dim = cond_dim
        self.n_cond_layers = n_cond_layers
        self.vision_n_layers = vision_n_layers

        # Vision encoder (patchify + pos + optional transformer layers)
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)
        self.pos_drop = nn.Dropout(dropout)

        Ht, Wt = image_size
        ph, pw = Ht // patch_size, Wt // patch_size
        self.num_patches = ph * pw
        if use_learned_img_pos:
            self.img_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
            nn.init.normal_(self.img_pos_embed, mean=0.0, std=0.02)
            self.register_parameter('img_pos_buffer', None)
        else:
            pos = _build_2d_sincos_pos_embed(embed_dim, ph, pw)  # [P, C]
            self.register_buffer('img_pos_buffer', pos.unsqueeze(0), persistent=False)
            self.img_pos_embed = None

        if vision_n_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(mlp_ratio * embed_dim),
                dropout=attn_dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.vision_encoder = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=vision_n_layers)
        else:
            self.vision_encoder = None

        # Waypoint input embedding
        self.input_emb = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Goal embedding (2D -> token)
        self.goal_emb = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.goal_pos_emb = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.goal_pos_emb, mean=0.0, std=0.02)
        # Learned positional encoding for K targets (and optional +1 time token when used as token)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_waypoints + (0 if time_as_cond else (0 if mode in ('encoder_decoder', 'decoder') else 1)), embed_dim))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        # Time embedding
        self.time_emb = SinusoidalPosEmb(embed_dim)
        # Optional cond obs embedding
        self.cond_obs_emb = None
        if self.obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, embed_dim)

        # Build encoder/decoder
        if mode == 'encoder':
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(mlp_ratio * embed_dim),
                dropout=attn_dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=depth)
            self.decoder = None
        else:
            # optional cond encoder
            if n_cond_layers > 0:
                cond_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(mlp_ratio * embed_dim),
                    dropout=attn_dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                )
                self.cond_encoder = nn.TransformerEncoder(encoder_layer=cond_layer, num_layers=n_cond_layers)
            else:
                # lightweight MLP fallback
                self.cond_encoder = nn.Sequential(
                    nn.Linear(embed_dim, 4 * embed_dim),
                    nn.Mish(),
                    nn.Linear(4 * embed_dim, embed_dim),
                )
            dec_layer = nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(mlp_ratio * embed_dim),
                dropout=attn_dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer=dec_layer, num_layers=depth)
            self.encoder = None

        # attention masks
        if self.causal_attn and self.mode != 'encoder':
            T = num_waypoints
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
            # nn.Transformer uses additive mask with -inf for masked positions
            mask = mask.float().masked_fill(mask, float('-inf')).masked_fill(~mask, 0.0)
            self.register_buffer('mask', mask)
        else:
            self.mask = None

        # output head
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 2)

        # init weights (best practices from example)
        self.apply(self._init_weights)

    # initialization adapted from model_from_example best practices
    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            # init proj weights if present
            for name in ['in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']:
                w = getattr(module, name, None)
                if w is not None:
                    nn.init.normal_(w, mean=0.0, std=0.02)
            for name in ['in_proj_bias', 'bias_k', 'bias_v']:
                b = getattr(module, name, None)
                if b is not None:
                    nn.init.zeros_(b)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        # pos embeddings already initialized where defined

    def get_optim_groups(self, weight_decay: float = 1e-3):
        decay = set()
        no_decay = set()
        whitelist = (nn.Linear, nn.MultiheadAttention)
        blacklist = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith('bias') or pn.startswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist):
                    no_decay.add(fpn)
        # special params
        no_decay.add('pos_emb')
        no_decay.add('_dummy_variable')
        if hasattr(self, 'img_pos_embed') and self.img_pos_embed is not None:
            no_decay.add('img_pos_embed')
        if hasattr(self, 'goal_pos_emb'):
            no_decay.add('goal_pos_emb')
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter = decay & no_decay
        union = decay | no_decay
        assert len(inter) == 0, f"parameters in both decay/no_decay: {inter}"
        assert len(param_dict.keys() - union) == 0, f"parameters not in any set: {param_dict.keys() - union}"
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return optim_groups

    def configure_optimizers(self, learning_rate: float = 1e-4, weight_decay: float = 1e-3, betas=(0.9, 0.95)):
        groups = self.get_optim_groups(weight_decay=weight_decay)
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)

    def _forward_vision(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to vision tokens (called once during inference).

        Args:
            image: [B, C, H, W] input image

        Returns:
            vision_tokens: [B, P, C] encoded vision tokens
        """
        B, _, H, W = image.shape
        if (H, W) != self.image_size:
            image = F.interpolate(image, size=self.image_size, mode='bilinear', align_corners=False)
        img_tokens, (ph, pw) = self.patch_embed(image)
        P = ph * pw
        if self.img_pos_embed is not None:
            pos = self.img_pos_embed
        else:
            pos = self.img_pos_buffer
            if pos.shape[1] != P:
                # recompute if needed (should not happen with fixed config)
                pos = _build_2d_sincos_pos_embed(self.embed_dim, ph, pw).to(img_tokens.device).unsqueeze(0)
        img_tokens = self.pos_drop(img_tokens + pos)
        if self.vision_encoder is not None:
            img_tokens = self.vision_encoder(img_tokens)

        return img_tokens  # [B, P, C]

    def _forward_goal(self, goal: torch.Tensor) -> torch.Tensor:
        """Encode goal (x,y) into a single token, reused across denoising steps.

        Args:
            goal: [B, 2] goal in local frame

        Returns:
            goal_token: [B, 1, C]
        """
        if not hasattr(self, 'goal_emb'):
            raise AttributeError("Model was not initialized with goal embedding components")
        g = self.goal_emb(goal).unsqueeze(1)  # [B,1,C]
        # add a learned positional embedding for the goal token
        g = self.pos_drop(g + self.goal_pos_emb)
        return g
    
    def _denoise(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        vision_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise given pre-encoded vision tokens and noisy sample.

        Args:
            sample: [B, K, 3] noisy waypoint sample
            timestep: diffusion timestep (scalar, [B], or [B, 1])
            vision_tokens: [B, P, C] pre-encoded vision tokens from _forward_vision()
            goal_tokens: [B, 1, C] pre-encoded goal tokens from _forward_goal()
            cond: optional [B, cond_dim] observation conditioning

        Returns:
            noise_pred: [B, K, 3] predicted noise
        """
        B = sample.shape[0]
        K = self.num_waypoints
        if sample.shape[1] != K:
            raise ValueError(f"Expected {K} waypoints, got {sample.shape[1]}")

        # time embedding
        t = timestep
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.long, device=sample.device)
        if t.ndim == 0:
            t = t[None].to(sample.device)
        t = t.expand(B)
        time_tok = self.time_emb(t).unsqueeze(1)  # [B,1,C]

        # tokens
        img_tokens = vision_tokens  # use pre-encoded vision tokens
        tgt_tokens = self.input_emb(sample)  # [B,K,C]

        if self.mode == 'encoder':
            seq: List[torch.Tensor] = []
            if self.time_as_cond:
                seq.append(time_tok)
            else:
                # add time as additive context to targets
                tgt_tokens = tgt_tokens + time_tok
            seq.append(img_tokens)
            # optional goal token as part of sequence
            seq.append(goal_tokens)
            # add positional enc to target tokens
            tgt = self.pos_drop(tgt_tokens + self.pos_emb[:, :tgt_tokens.shape[1], :])
            seq.append(tgt)
            x = torch.cat(seq, dim=1)  # [B, S+K, C]
            x = self.encoder(x)
            x = x[:, -K:, :]  # take the last K target positions
        else:
            # build memory (cond) tokens
            mem_list: List[torch.Tensor] = []
            if self.time_as_cond:
                mem_list.append(time_tok)
            else:
                tgt_tokens = tgt_tokens + time_tok
            # optional obs cond tokens
            if self.obs_as_cond and (cond is not None):
                mem_list.append(self.cond_obs_emb(cond))
            # optional goal token as memory
            mem_list.append(goal_tokens)
            mem_list.append(img_tokens)
            memory = torch.cat(mem_list, dim=1) if len(mem_list) > 1 else mem_list[0]
            if isinstance(self.cond_encoder, nn.Sequential) or isinstance(self.cond_encoder, nn.TransformerEncoder):
                memory = self.cond_encoder(memory)
            # target tokens + pos
            tgt = self.pos_drop(tgt_tokens + self.pos_emb[:, :tgt_tokens.shape[1], :])
            x = self.decoder(tgt=tgt, memory=memory, tgt_mask=self.mask, memory_mask=None)

        x = self.ln_f(x)
        x = self.head(x)
        return x

    def forward(
        self,
        image: torch.Tensor,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        goal: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for training (encodes vision and denoises in one call).

        Args:
            image: [B, C, H, W] input image
            sample: [B, K, 3] noisy waypoint sample
            timestep: diffusion timestep
            cond: optional observation conditioning

        Returns:
            noise_pred: [B, K, 3] predicted noise
        """
        vision_tokens = self._forward_vision(image) # [B, P, C]
        goal_tokens = self._forward_goal(goal) # [B, 1, C]
        return self._denoise(sample, timestep, vision_tokens, goal_tokens, cond)


def build_model(
    image_size: Tuple[int, int],
    patch_size: int = 8,
    in_chans: int = 1,
    embed_dim: int = 256,
    depth: int = 6,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    num_waypoints: int = 5,
    dropout: float = 0.1,
    attn_dropout: float = 0.1,
    use_learned_img_pos: bool = False,
    mode: str = 'encoder_decoder',
    causal_attn: bool = False,
    time_as_cond: bool = True,
    obs_as_cond: bool = False,
    cond_dim: int = 0,
    n_cond_layers: int = 0,
    vision_n_layers: int = 0,
):
    return DiffusionWaypointTransformer(
        image_size=image_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_waypoints=num_waypoints,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attn_dropout=attn_dropout,
        use_learned_img_pos=use_learned_img_pos,
        mode=mode,
        causal_attn=causal_attn,
        time_as_cond=time_as_cond,
        obs_as_cond=obs_as_cond,
        cond_dim=cond_dim,
        n_cond_layers=n_cond_layers,
        vision_n_layers=vision_n_layers,
    )
