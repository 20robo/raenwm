# train_pose_probe.py
import os, sys, math, time, yaml, argparse, logging, pickle, random, json
from datetime import datetime
from typing import Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# ---- make RAE/src importable (same pattern as train_rae.py) ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAE_SRC = os.path.join(PROJECT_ROOT, "RAE", "src")
if RAE_SRC not in sys.path:
    sys.path.append(RAE_SRC)

import wandb
import tqdm
import numpy as np
from distributed import init_distributed
from datasets import TrainingDataset, EvalDataset
from misc import transform, unnormalize_data, get_data_path, normalize_data
from RAE.src.utils.model_utils import instantiate_from_config
from RAE.src.utils.train_utils import parse_configs

from probe import LinearForwardDynamicsProbe


def create_logger(log_dir: str):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{log_dir}/log.txt")]
        )
        return logging.getLogger("pose_probe")
    logger = logging.getLogger("pose_probe")
    logger.addHandler(logging.NullHandler())
    return logger


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def r2_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    target_mean = target.mean()
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((pred - target) ** 2)
    return 1.0 - ss_res / (ss_tot + eps)


def yaw_similarity(pred: torch.Tensor, target_yaw: torch.Tensor, yaw_mode: str) -> torch.Tensor:
    if yaw_mode == "sincos":
        sc_p = F.normalize(pred[:, 2:4], dim=-1)
        sc_t = torch.stack([torch.sin(target_yaw), torch.cos(target_yaw)], dim=-1)
        return torch.sum(sc_p * sc_t, dim=-1).mean()
    yaw_p = pred[:, 2]
    return torch.cos(wrap_to_pi(yaw_p - target_yaw)).mean()


def compose_delta_se2(delta_seq: torch.Tensor) -> torch.Tensor:
    if delta_seq.dim() != 3 or delta_seq.shape[-1] < 3:
        raise ValueError(f"Expected delta_seq as (B,K,>=3), got {tuple(delta_seq.shape)}")

    dx = delta_seq[..., 0]
    dy = delta_seq[..., 1]
    dth = delta_seq[..., 2]

    x = torch.zeros(dx.shape[0], device=delta_seq.device, dtype=delta_seq.dtype)
    y = torch.zeros_like(x)
    th = torch.zeros_like(x)

    for k in range(dx.shape[1]):
        c = torch.cos(th)
        s = torch.sin(th)
        x = x + c * dx[:, k] - s * dy[:, k]
        y = y + s * dx[:, k] + c * dy[:, k]
        th = th + dth[:, k]

    two_pi = 2.0 * math.pi
    th = th - two_pi * torch.floor((th + math.pi) / two_pi)

    return torch.stack([x, y, th], dim=-1)


@torch.no_grad()
def encode_with_rae(rae, imgs: torch.Tensor, bfloat16: bool = True) -> torch.Tensor:
    x = imgs * 0.5 + 0.5
    with torch.amp.autocast("cuda", enabled=bfloat16, dtype=torch.bfloat16):
        lat = rae.encode(x)
    return lat.float()


def build_encoder(config: Dict[str, Any], device: torch.device):
    enc = config.get("encoder", {})
    enc_type = str(enc.get("type", "rae")).lower()

    if enc_type == "rae":
        rae_cfg_path = str(enc.get("rae_config_path", ""))
        if not rae_cfg_path:
            rae_cfg_path = config.get("config_path", "RAE/configs/stage1/pretrained/DINOv2-B.yaml")
        rae_config, *_ = parse_configs(rae_cfg_path)
        model = instantiate_from_config(rae_config).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        in_channels = int(getattr(model, "latent_dim", 768))

        def encode_fn(imgs: torch.Tensor, bfloat16: bool) -> torch.Tensor:
            return encode_with_rae(model, imgs, bfloat16=bfloat16)

        return enc_type, model, encode_fn, in_channels

    if enc_type == "vae":
        from diffusers.models import AutoencoderKL

        vae_name = str(enc.get("vae_name", "stabilityai/sd-vae-ft-ema"))
        vae_scale = float(enc.get("vae_scale", 0.18215))
        use_mean = bool(enc.get("vae_use_mean", True))

        model = AutoencoderKL.from_pretrained(vae_name).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        in_channels = 4

        def encode_fn(imgs: torch.Tensor, bfloat16: bool) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=bfloat16, dtype=torch.bfloat16):
                dist_out = model.encode(imgs).latent_dist
                lat = dist_out.mean if use_mean else dist_out.sample()
                lat = lat * vae_scale
            return lat.float()

        return enc_type, model, encode_fn, in_channels

    if enc_type == "resnet50":
        from torchvision.models import resnet50
        try:
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT
            model = resnet50(weights=weights).to(device).eval()
        except Exception:
            model = resnet50(weights=None).to(device).eval()

        for p in model.parameters():
            p.requires_grad_(False)

        in_channels = 2048

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        def encode_fn(imgs: torch.Tensor, bfloat16: bool) -> torch.Tensor:
            x = (imgs * 0.5 + 0.5).clamp(0, 1)
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = (x - mean.to(dtype=x.dtype)) / std.to(dtype=x.dtype)
            with torch.amp.autocast("cuda", enabled=bfloat16, dtype=torch.bfloat16):
                x = model.conv1(x)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
            return x.float()

        return enc_type, model, encode_fn, in_channels

    if enc_type == "dinov2_cls":
        repo = str(enc.get("dino_repo", "facebookresearch/dinov2"))
        name = str(enc.get("dino_name", "dinov2_vitb14"))
        model = torch.hub.load(repo, name).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        in_channels = int(getattr(model, "embed_dim", 768))

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        def encode_fn(imgs: torch.Tensor, bfloat16: bool) -> torch.Tensor:
            x = (imgs * 0.5 + 0.5).clamp(0, 1)
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = (x - mean.to(dtype=x.dtype)) / std.to(dtype=x.dtype)
            with torch.amp.autocast("cuda", enabled=bfloat16, dtype=torch.bfloat16):
                cls = None
                if hasattr(model, "forward_features"):
                    feats = model.forward_features(x)
                    if isinstance(feats, dict):
                        cls = feats.get("x_norm_clstoken", None)
                        if cls is None:
                            cls = feats.get("x_norm_cls_token", None)
                        if cls is None:
                            cls = feats.get("x_prenorm", None)
                            if cls is not None and cls.dim() == 3:
                                cls = cls[:, 0]
                    elif isinstance(feats, torch.Tensor):
                        cls = feats
                if cls is None:
                    cls = model(x)

            if cls.dim() > 2:
                cls = cls.reshape(cls.shape[0], -1)
            return cls.to(dtype=torch.float32).view(cls.shape[0], cls.shape[1], 1, 1)

        return enc_type, model, encode_fn, in_channels

    if enc_type == "pixel":
        pixel_size = enc.get("pixel_size", None)
        in_channels = 3

        def encode_fn(imgs: torch.Tensor, bfloat16: bool) -> torch.Tensor:
            x = imgs
            if pixel_size is not None:
                s = int(pixel_size)
                x = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
            return x.float()

        return enc_type, None, encode_fn, in_channels

    raise ValueError(f"Unknown encoder.type={enc_type}")


def apply_latent_aug(x: torch.Tensor, sigma_min: float, sigma_max: float, token_dropout_p: float) -> torch.Tensor:
    if sigma_max > 0:
        sigma = torch.empty((x.shape[0], 1, 1, 1), device=x.device).uniform_(sigma_min, sigma_max)
        x = x + sigma * torch.randn_like(x)

    if token_dropout_p > 0 and x.dim() == 4:
        keep = (torch.rand((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device) > token_dropout_p).float()
        x = x * keep
    return x


def _img_tensor_to_pil(x_chw: torch.Tensor) -> Image.Image:
    x = (x_chw.detach().cpu() * 0.5 + 0.5).clamp(0, 1)
    x = (x * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(x, mode="RGB")


def _make_probe_viz_panel(
    curr: torch.Tensor,
    goal: torch.Tensor,
    pred: torch.Tensor,
    tgt: torch.Tensor,
    action_stats: Dict[str, torch.Tensor],
    yaw_mode: str,
    max_items: int = 4,
) -> Image.Image:
    n = min(int(max_items), int(curr.shape[0]))
    if n <= 0:
        raise ValueError("max_items must be >= 1")

    font = ImageFont.load_default()

    with torch.no_grad():
        dxdy_p = unnormalize_data(pred[:n, :2].detach(), action_stats)
        dxdy_t = unnormalize_data(tgt[:n, :2].detach(), action_stats)
        err_xy = torch.norm(dxdy_p - dxdy_t, dim=-1)

        yaw_p = decode_yaw(pred[:n].detach(), yaw_mode=yaw_mode)
        yaw_t = tgt[:n, 2].detach()
        err_yaw = wrap_to_pi(yaw_p - yaw_t).abs() * (180.0 / math.pi)

    rows = []
    for i in range(n):
        ci = _img_tensor_to_pil(curr[i])
        gi = _img_tensor_to_pil(goal[i])
        pair = Image.new("RGB", (ci.width + gi.width, ci.height))
        pair.paste(ci, (0, 0))
        pair.paste(gi, (ci.width, 0))

        gt_dx, gt_dy = float(dxdy_t[i, 0].cpu()), float(dxdy_t[i, 1].cpu())
        pr_dx, pr_dy = float(dxdy_p[i, 0].cpu()), float(dxdy_p[i, 1].cpu())
        gt_y = float((yaw_t[i].cpu() * (180.0 / math.pi)))
        pr_y = float((yaw_p[i].cpu() * (180.0 / math.pi)))
        exy = float(err_xy[i].cpu())
        ey = float(err_yaw[i].cpu())

        text_lines = [
            f"GT  dxdy=({gt_dx:+.2f},{gt_dy:+.2f}) yaw={gt_y:+.1f}deg",
            f"PR  dxdy=({pr_dx:+.2f},{pr_dy:+.2f}) yaw={pr_y:+.1f}deg",
            f"ERR xy={exy:.2f} step, yaw={ey:.1f} deg",
        ]

        draw = ImageDraw.Draw(pair)
        rect_h = 4 + len(text_lines) * 12
        draw.rectangle((0, 0, pair.width, rect_h), fill=(0, 0, 0))
        y0 = 2
        for line in text_lines:
            draw.text((4, y0), line, fill=(255, 255, 255), font=font)
            y0 += 12

        rows.append(pair)

    panel = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)))
    for i, r in enumerate(rows):
        panel.paste(r, (0, i * rows[0].height))
    return panel


class SequentialTrainingDataset(TrainingDataset):
    """
    A variant of TrainingDataset that samples ALL future goals sequentially 
    up to max_dist_cat, instead of random sampling.
    This improves data utilization for probe training.
    """
    def _load_index(self, predefined_index) -> None:
        """
        Override to use a separate cache file for sequential training to avoid conflict
        with the standard TrainingDataset cache.
        """
        if predefined_index:
            super()._load_index(predefined_index)
            return

        if dist.get_rank() == 0:
            print("****** Building/Loading NON PREDEFINED index (Sequential) ... ******")
        
        # Use a distinct filename suffix
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_n{self.context_size}_len_traj_pred_{self.len_traj_pred}_sequential.pkl",
        )
        
        if os.path.exists(index_to_data_path):
             with open(index_to_data_path, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)
        else:
            self.index_to_data, self.goals_index = self._build_index()
            # Only rank 0 should write, but here we assume simple execution or handled by race condition (standard in this codebase?)
            # The original code didn't lock, so we follow suit, but typically only rank 0 builds or all build.
            # In DDP, usually one should build and others wait. But for simplicity we just build.
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _build_index(self, use_tqdm: bool = False):
        """Build an index consisting of tuples (trajectory name, time, min_goal_dist, max_goal_dist)."""
        samples_index = []
        goals_index = []

        # Only show tqdm on rank 0
        disable_tqdm = (not use_tqdm) or (dist.is_initialized() and dist.get_rank() != 0)

        for traj_name in tqdm.tqdm(self.traj_names, disable=disable_tqdm, dynamic_ncols=True, desc=f"Building Index {self.dataset_name}"):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])

            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = max(self.context_size - 1, -int(self.min_dist_cat))
            end_time = min(traj_len - int(self.len_traj_pred), traj_len - int(self.max_dist_cat))

            for curr_time in range(begin_time, end_time, self.traj_stride):
                max_goal_distance = min(int(self.max_dist_cat), traj_len - curr_time - 1)
                min_goal_distance = max(int(self.min_dist_cat), -curr_time)
                samples_index.append((traj_name, curr_time, min_goal_distance, max_goal_distance))

        return samples_index, goals_index

    def __getitem__(self, i: int):
        try:
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
            
            # Deterministic sequential sampling: all offsets from min to max
            # This ensures we get exactly (max_dist_cat - min_dist_cat + 1) goals
            goal_offset = np.arange(min_goal_dist, max_goal_dist + 1)
            
            goal_time = (curr_time + goal_offset).astype('int')
            rel_time = (goal_offset).astype('float')/(128.) # TODO: refactor

            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            # Combine context + all goals
            # Note: TrainingDataset stacks them.
            context = [(f_curr, t) for t in context_times] + [(f_curr, t) for t in goal_time]

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])
            
            # We don't really use context_rel_paths for training, but keep structure
            context_rel_paths = [os.path.join(self.dataset_name, f, f"{t}.jpg") for f, t in context]

            # Load other trajectory data
            curr_traj_data = self._get_trajectory(f_curr)

            # Compute actions
            _, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
            goal_pos[:, :2] = normalize_data(goal_pos[:, :2], self.ACTION_STATS)

            return (
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
                torch.as_tensor(rel_time, dtype=torch.float32),
                context_rel_paths,
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            # Raise exception to fail fast if something is wrong
            raise Exception(e)


def build_dataloaders(config: Dict[str, Any], device: torch.device, seed: int = 0):
    train_sets, test_sets = [], []

    probe_cfg = config.get("pose_probe", {})
    task = str(probe_cfg.get("task", "pose")).lower()
    action_stats_mode = str(probe_cfg.get("action_stats_mode", "global")).lower()

    for dataset_name, dcfg in config["datasets"].items():
        for split in ["train", "test"]:
            if split not in dcfg:
                continue
            
            if task in {"forward", "forward_dynamics", "latent_forward", "latent_fwd", "fwd"}:
                DatasetClass = EvalDataset
            else:
                DatasetClass = TrainingDataset

            ds_kwargs = {
                "data_folder": dcfg["data_folder"],
                "data_split_folder": dcfg[split],
                "dataset_name": dataset_name,
                "image_size": (int(config["image_size"]), int(config["image_size"])),
                "min_dist_cat": int(config["distance"]["min_dist_cat"]),
                "max_dist_cat": int(config["distance"]["max_dist_cat"]),
                "len_traj_pred": int(config["len_traj_pred"]),
                "context_size": int(config["context_size"]),
                "normalize": bool(config.get("normalize", True)),
                "goals_per_obs": int(dcfg.get("goals_per_obs", 1)),
                "transform": transform,
                "predefined_index": None,
                "traj_stride": 1,
            }
            if DatasetClass is EvalDataset:
                ds_kwargs["traj_names"] = "traj_names.txt"

            ds = DatasetClass(**ds_kwargs)

            # Optionally override dx/dy normalization bounds to match probe-local distance range.
            if action_stats_mode in {"dist_cat", "local"}:
                local_min = float(config["distance"]["min_dist_cat"])
                local_max = float(config["distance"]["max_dist_cat"])
                ds.ACTION_STATS = {
                    "min": np.array([[local_min, local_min]], dtype=np.float32),
                    "max": np.array([[local_max, local_max]], dtype=np.float32),
                }

            if split == "train":
                train_sets.append(ds)
            else:
                test_sets.append(ds)

    train_dataset = ConcatDataset(train_sets)
    test_dataset = ConcatDataset(test_sets) if len(test_sets) > 0 else None

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        drop_last=True,
        seed=int(seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        sampler=train_sampler,
        shuffle=False,
        num_workers=int(config.get("num_workers", 8)),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    test_loader = None
    if test_dataset is not None:
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False,
            seed=int(seed),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(config["batch_size"]),
            sampler=test_sampler,
            shuffle=False,
            num_workers=int(config.get("num_workers", 8)),
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )

    # ACTION_STATS for success/metric
    if action_stats_mode in {"dist_cat", "local"}:
        local_min = float(config["distance"]["min_dist_cat"])
        local_max = float(config["distance"]["max_dist_cat"])
        action_stats = {
            "min": torch.tensor([local_min, local_min], device=device).view(1, 2),
            "max": torch.tensor([local_max, local_max], device=device).view(1, 2),
        }
    else:
        with open(os.path.join(PROJECT_ROOT, "config", "data_config.yaml"), "r") as f:
            dc = yaml.safe_load(f)
        action_stats = {
            "min": torch.tensor(dc["action_stats"]["min"], device=device).view(1, 2),
            "max": torch.tensor(dc["action_stats"]["max"], device=device).view(1, 2),
        }

    return train_loader, train_sampler, test_loader, action_stats


@torch.no_grad()
def eval_probe(
    probe_ddp: DDP,
    encode_fn,
    loader: DataLoader,
    action_stats: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    device: torch.device,
    bfloat16: bool,
    max_batches_override: Optional[int] = None,
):
    probe_cfg = config["pose_probe"]
    task = str(probe_cfg.get("task", "forward_dynamics")).lower()
    if task not in {"forward", "forward_dynamics", "latent_forward", "latent_fwd", "fwd"}:
        raise ValueError("train_pose_probe.py only supports task=forward_dynamics with LinearForwardDynamicsProbe")
    yaw_mode = probe_cfg["yaw_mode"]
    avoid_zero = bool(probe_cfg.get("avoid_zero_offset", True))
    if max_batches_override is None:
        max_batches = int(config.get("eval_num_batches", 50))
    else:
        max_batches = int(max_batches_override)

    count = torch.tensor(0.0, device=device)

    sum_dx_t = torch.tensor(0.0, device=device)
    sum_dx_t2 = torch.tensor(0.0, device=device)
    sum_dx_sse = torch.tensor(0.0, device=device)

    sum_dy_t = torch.tensor(0.0, device=device)
    sum_dy_t2 = torch.tensor(0.0, device=device)
    sum_dy_sse = torch.tensor(0.0, device=device)

    sum_yaw_sim = torch.tensor(0.0, device=device)

    probe_ddp.eval()

    for bidx, batch in enumerate(loader):
        if bidx >= max_batches:
            break
        x, y, rel_t = batch[:3]  # (B,T,3,H,W), (B,G,3), (B,G)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        rel_t = rel_t.to(device, non_blocking=True)

        B, T = x.shape[:2]
        G = y.shape[1]
        ctx = int(config["context_size"])
        assert T == ctx + G

        curr = x[:, ctx - 1]          # (B,3,H,W)
        goal = x[:, ctx:]             # (B,G,3,H,W)

        curr = curr.unsqueeze(1).expand(B, G, *curr.shape[1:]).reshape(B * G, *curr.shape[1:])
        goal = goal.reshape(B * G, *goal.shape[2:])
        tgt = y.reshape(B * G, 3)
        rel_flat = rel_t.reshape(B * G)

        if avoid_zero:
            m = rel_flat.abs() > 1e-12
            if m.sum() == 0:
                continue
            curr, goal, tgt, rel_flat = curr[m], goal[m], tgt[m], rel_flat[m]

        curr_lat = encode_fn(curr, bfloat16=bfloat16)
        goal_lat = encode_fn(goal, bfloat16=bfloat16)

        if bool(config["pose_probe"].get("use_rel_t", False)):
            pred = probe_ddp(curr_lat, goal_lat, rel_t=rel_flat)
        else:
            pred = probe_ddp(curr_lat, goal_lat)


        dxdy_p = unnormalize_data(pred[:, :2], action_stats)
        dxdy_t = unnormalize_data(tgt[:, :2], action_stats)

        yaw_t = tgt[:, 2]

        n = torch.tensor(float(dxdy_t.shape[0]), device=device)
        count += n

        dx_t = dxdy_t[:, 0]
        dy_t = dxdy_t[:, 1]
        dx_p = dxdy_p[:, 0]
        dy_p = dxdy_p[:, 1]

        sum_dx_t += dx_t.sum()
        sum_dx_t2 += (dx_t ** 2).sum()
        sum_dx_sse += ((dx_p - dx_t) ** 2).sum()

        sum_dy_t += dy_t.sum()
        sum_dy_t2 += (dy_t ** 2).sum()
        sum_dy_sse += ((dy_p - dy_t) ** 2).sum()

        sum_yaw_sim += yaw_similarity(pred, yaw_t, yaw_mode=yaw_mode) * n

    for t in [
        count,
        sum_dx_t, sum_dx_t2, sum_dx_sse,
        sum_dy_t, sum_dy_t2, sum_dy_sse,
        sum_yaw_sim,
    ]:
        all_reduce_sum(t)

    eps = 1e-9
    n = count + eps

    sst_dx = sum_dx_t2 - (sum_dx_t ** 2) / n
    sst_dy = sum_dy_t2 - (sum_dy_t ** 2) / n

    out = {
        "eval/r2_dx_step": float((1.0 - sum_dx_sse / (sst_dx + eps)).cpu()),
        "eval/r2_dy_step": float((1.0 - sum_dy_sse / (sst_dy + eps)).cpu()),
        "eval/yaw_sim": float((sum_yaw_sim / n).cpu()),
    }

    return out


@torch.no_grad()
def eval_forward_dynamics_probe(
    probe_ddp: DDP,
    encode_fn,
    loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    bfloat16: bool,
    max_batches_override: Optional[int] = None,
    spatial_shuffle: bool = False,
    action_override: str = "none",
    action_random_mode: str = "shuffle",
):
    max_batches = int(config.get("eval_num_batches", 50)) if max_batches_override is None else int(max_batches_override)

    count = torch.tensor(0.0, device=device)
    sum_t = torch.tensor(0.0, device=device)
    sum_t2 = torch.tensor(0.0, device=device)
    sum_sse = torch.tensor(0.0, device=device)

    probe_ddp.eval()

    for bidx, batch in enumerate(loader):
        if bidx >= max_batches:
            break

        _, obs, pred, delta = batch[:4]
        obs = obs.to(device, non_blocking=True)
        pred = pred.to(device, non_blocking=True)
        delta = delta.to(device, non_blocking=True)

        probe_cfg = config["pose_probe"]
        k = int(probe_cfg.get("forward_step", 2))
        k = max(1, k)
        k = min(k, int(pred.shape[1]), int(delta.shape[1]))

        x_t = obs[:, -1]
        x_tpk = pred[:, k - 1]

        if k == 1:
            a_t = delta[:, 0]
        else:
            a_t = compose_delta_se2(delta[:, :k, :3])

        a_mode = str(action_override).lower().strip()
        if a_mode in {"", "none", "normal"}:
            pass
        elif a_mode in {"shuffle", "shuffled", "permute"}:
            if int(a_t.shape[0]) > 1:
                perm = torch.randperm(int(a_t.shape[0]), device=a_t.device)
                a_t = a_t[perm]
        else:
            raise ValueError(f"Unknown action_override={action_override}. Supported: none|shuffle")

        z_t = encode_fn(x_t, bfloat16=bfloat16)
        z_tpk = encode_fn(x_tpk, bfloat16=bfloat16)

        z_hat = probe_ddp(z_t, a_t, spatial_shuffle=bool(spatial_shuffle))

        t = z_tpk.reshape(-1)
        p = z_hat.reshape(-1)

        n = torch.tensor(float(t.numel()), device=device)
        count += n
        sum_t += t.sum()
        sum_t2 += (t ** 2).sum()
        sum_sse += ((p - t) ** 2).sum()

    for t in [count, sum_t, sum_t2, sum_sse]:
        all_reduce_sum(t)

    eps = 1e-9
    n = count + eps
    sst = sum_t2 - (sum_t ** 2) / n

    nmse = sum_sse / (sst + eps)
    r2 = 1.0 - sum_sse / (sst + eps)

    return {
        "eval/nmse_z": float(nmse.cpu()),
        "eval/r2_z": float(r2.cpu()),
    }


def _parse_csv_list(s: str):
    if s is None:
        return []
    parts = [p.strip() for p in str(s).split(",")]
    return [p for p in parts if len(p) > 0]


def _select_best_ckpt_path(ckpt_dir: str) -> Optional[str]:
    if not ckpt_dir:
        return None
    for name in [
        "best_forward_dynamics.pth",
        "best_pose.pth",
        "best.pth",
    ]:
        p = os.path.join(ckpt_dir, name)
        if os.path.isfile(p):
            return p
    return None


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--global-seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--bfloat16", type=int, default=1)
    p.add_argument("--torch-compile", type=int, default=None)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--encoder-type", type=str, default=None)
    p.add_argument("--encoder-rae-config", type=str, default=None)
    p.add_argument("--forward-probe-type", type=str, default=None)
    p.add_argument("--forward-step", type=int, default=None)
    p.add_argument("--fixed-lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--run-suffix", type=str, default=None)
    p.add_argument("--only-spatial-random-test", type=int, default=0)
    p.add_argument("--test-only", type=int, default=0)
    p.add_argument("--exp-dir", type=str, default="")
    p.add_argument("--test-action-modes", type=str, default="")
    p.add_argument("--action-random-mode", type=str, default="shuffle")
    return p.parse_args()


def main():
    assert torch.cuda.is_available()
    _, rank, gpu, _ = init_distributed()
    device = torch.device(f"cuda:{gpu}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = get_args()

    seed = int(args.global_seed) * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    exp_dir_override = str(args.exp_dir).strip() if args.exp_dir is not None else ""

    if not exp_dir_override:
        if args.encoder_type is not None:
            enc_cfg = config.get("encoder", {})
            enc_cfg["type"] = str(args.encoder_type)
            config["encoder"] = enc_cfg
        if args.encoder_rae_config is not None:
            cfg_path = str(args.encoder_rae_config)
            config["config_path"] = cfg_path
            enc_cfg = config.get("encoder", {})
            enc_cfg["rae_config_path"] = cfg_path
            config["encoder"] = enc_cfg
        if args.forward_probe_type is not None:
            pp_cfg = config.get("pose_probe", {})
            pp_cfg["forward_probe_type"] = str(args.forward_probe_type)
            config["pose_probe"] = pp_cfg
        if args.forward_step is not None:
            pp_cfg = config.get("pose_probe", {})
            pp_cfg["forward_step"] = int(args.forward_step)
            config["pose_probe"] = pp_cfg
        if args.fixed_lr is not None:
            pp_cfg = config.get("pose_probe", {})
            lr_value = float(args.fixed_lr)
            pp_cfg["lr"] = lr_value
            pp_cfg["min_lr"] = lr_value
            pp_cfg["warmup_steps"] = 0
            config["pose_probe"] = pp_cfg
        if args.batch_size is not None:
            config["batch_size"] = int(args.batch_size)
        if args.run_suffix is not None and len(str(args.run_suffix)) > 0:
            base_name = str(config.get("run_name", "probe"))
            config["run_name"] = base_name + "_" + str(args.run_suffix)

    # dirs
    if exp_dir_override:
        exp_dir = exp_dir_override
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        if rank == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = _select_best_ckpt_path(ckpt_dir)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location={"cuda:0": f"cuda:{gpu}"})
            ckpt_cfg = ckpt.get("config", None)
            if isinstance(ckpt_cfg, dict):
                config = ckpt_cfg
                config["results_dir"] = os.path.dirname(exp_dir)
                config["run_name"] = os.path.basename(exp_dir)
            if not args.resume:
                args.resume = ckpt_path
    else:
        os.makedirs(config["results_dir"], exist_ok=True)
        exp_dir = os.path.join(config["results_dir"], config["run_name"])
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        if rank == 0:
            os.makedirs(ckpt_dir, exist_ok=True)

    logger = create_logger(exp_dir)

    # wandb
    if rank == 0 and config.get("wandb", {}).get("enabled", False):
        wb = config["wandb"]
        run_name = wb.get("run_name", None) or (config["run_name"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        wandb.init(
            project=wb.get("project", "nwm-probe"),
            entity=wb.get("entity", None),
            name=run_name,
            tags=wb.get("tags", None),
            notes=wb.get("notes", None),
            config=config,
        )

    enc_type, encoder_model, encode_fn, in_channels = build_encoder(config, device)

    train_loader, train_sampler, test_loader, action_stats = build_dataloaders(config, device, seed=int(args.global_seed))

    probe_cfg = config["pose_probe"]
    task = str(probe_cfg.get("task", "pose")).lower()

    probe_kwargs = {
        "in_channels": in_channels,
        "yaw_mode": str(probe_cfg.get("yaw_mode", "sincos")),
        "dropout": float(probe_cfg.get("dropout", 0.1)),
        "use_rel_t": bool(probe_cfg.get("use_rel_t", False)),
        "rel_t_dim": int(probe_cfg.get("rel_t_dim", 64)),
        "rel_t_dropout": float(probe_cfg.get("rel_t_dropout", 0.1)),
    }

    if "hidden_dim" in probe_cfg:
        probe_kwargs["hidden_dim"] = int(probe_cfg["hidden_dim"])

    probe_kwargs = {
        "in_channels": int(in_channels),
        "action_dim": int(probe_cfg.get("action_dim", 3)),
        "predict_residual": bool(probe_cfg.get("predict_residual", True)),
    }
    ProbeClass = LinearForwardDynamicsProbe


    probe = ProbeClass(**probe_kwargs).to(device)
    if args.torch_compile:
        probe = torch.compile(probe)
    probe_ddp = DDP(probe, device_ids=[gpu], find_unused_parameters=False)

    if rank == 0:
        n_total = sum(p.numel() for p in probe_ddp.module.parameters())
        n_train = sum(p.numel() for p in probe_ddp.module.parameters() if p.requires_grad)
        logger.info(f"Probe params: total={n_total:,} trainable={n_train:,}")

    opt = torch.optim.AdamW(
        probe_ddp.parameters(),
        lr=float(probe_cfg.get("lr", 1e-4)),
        weight_decay=float(probe_cfg.get("weight_decay", 0.01)),
    )

    forward_loss = str(probe_cfg.get("forward_loss", "mse")).lower()

    best_metric_default = "eval/r2_z"
    best_metric = str(probe_cfg.get("best_metric", best_metric_default))
    best_mode = str(probe_cfg.get("best_mode", "auto")).lower()
    if best_mode == "auto":
        if any(k in best_metric for k in ["r2", "yaw_sim", "success"]):
            best_mode = "max"
        else:
            best_mode = "min"

    global_step = 0
    best = -1e9 if best_mode == "max" else 1e9

    is_forward = True
    task_key = "forward_dynamics"
    best_fname = f"best_{task_key}.pth"

    auto_resume_path = os.path.join(ckpt_dir, best_fname)
    legacy_auto_resume_path = os.path.join(ckpt_dir, "best.pth")

    if not args.resume:
        should_resume = False
        chosen = None
        if rank == 0:
            if os.path.isfile(auto_resume_path):
                should_resume = True
                chosen = auto_resume_path
            elif os.path.isfile(legacy_auto_resume_path):
                should_resume = True
                chosen = legacy_auto_resume_path
        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor([1 if should_resume else 0], device=device, dtype=torch.int64)
            dist.broadcast(flag, src=0)
            should_resume = bool(int(flag.item()))
        if should_resume:
            args.resume = chosen or auto_resume_path
            if rank == 0:
                logger.info(f"Auto-resume from {args.resume}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location={"cuda:0": f"cuda:{gpu}"})
        probe_sd = ckpt.get("probe", {})
        ckpt_is_forward = ("A.weight" in probe_sd) or ("B.weight" in probe_sd) or (ckpt.get("probe_task") == "forward_dynamics")
        if ckpt_is_forward != is_forward:
            if rank == 0:
                logger.warning(f"Skip resume from {args.resume}: checkpoint task={'forward_dynamics' if ckpt_is_forward else 'pose'} != current task={task_key}.")
        else:
            probe_ddp.module.load_state_dict(probe_sd, strict=True)
            opt.load_state_dict(ckpt["opt"])
            global_step = int(ckpt.get("global_step", 0))
            if "best_metric_value" in ckpt:
                best = float(ckpt["best_metric_value"])
            elif "best_eval_mae_yaw_deg" in ckpt:
                best = float(ckpt["best_eval_mae_yaw_deg"])
            if rank == 0:
                logger.info(f"Resumed from {args.resume}, step={global_step}, best_metric={best_metric}, best={best:.6f}")

    # schedule
    if args.epochs is not None:
        max_steps = int(args.epochs) * max(1, len(train_loader))
    else:
        max_steps = int(probe_cfg.get("max_steps", 60000))

    if bool(args.only_spatial_random_test):
        max_steps = 0

    if bool(args.test_only):
        max_steps = 0

    warmup_steps = int(probe_cfg.get("warmup_steps", 2000))

    log_every = int(args.log_every) if args.log_every is not None else int(probe_cfg.get("log_every_steps", 100))
    eval_every = int(args.eval_every) if args.eval_every is not None else int(probe_cfg.get("eval_every_steps", 2000))
    save_every = int(args.ckpt_every) if args.ckpt_every is not None else int(probe_cfg.get("save_every_steps", 5000))

    grad_clip = float(probe_cfg.get("grad_clip_val", 1.0))

    sigma_min = float(probe_cfg.get("noise_sigma_min", 0.0))
    sigma_max = float(probe_cfg.get("noise_sigma_max", 0.08))
    tok_drop = float(probe_cfg.get("token_dropout_p", 0.10))
    avoid_zero = bool(probe_cfg.get("avoid_zero_offset", True))

    bfloat16 = bool(args.bfloat16)

    if rank == 0:
        logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
        logger.info(f"Experiment dir: {exp_dir}")
        logger.info(f"Train: max_steps={max_steps}, eval_every={eval_every}, save_every={save_every}")
        logger.info(f"Distance range: [{config['distance']['min_dist_cat']},{config['distance']['max_dist_cat']}], len_traj_pred={config['len_traj_pred']}")
        logger.info(f"Noise aug: sigma~U({sigma_min},{sigma_max}), token_dropout_p={tok_drop}")
        logger.info(f"Probe: pooling={probe_cfg.get('pooling')}, yaw_mode={probe_cfg.get('yaw_mode')}")

    # train loop
    probe_ddp.train()
    t0 = time.time()
    running_loss = 0.0
    running_n = 0

    while global_step < max_steps:
        train_sampler.set_epoch(global_step // max(1, len(train_loader)))
        for batch in train_loader:
            if global_step >= max_steps:
                break

            if task in {"forward", "forward_dynamics", "latent_forward", "latent_fwd", "fwd"}:
                _, obs, pred_frames, delta = batch[:4]
                obs = obs.to(device, non_blocking=True)
                pred_frames = pred_frames.to(device, non_blocking=True)
                delta = delta.to(device, non_blocking=True)

                k = int(probe_cfg.get("forward_step", 2))
                k = max(1, k)
                k = min(k, int(pred_frames.shape[1]), int(delta.shape[1]))

                x_t = obs[:, -1]
                x_tpk = pred_frames[:, k - 1]

                if k == 1:
                    a_t = delta[:, 0]
                else:
                    a_t = compose_delta_se2(delta[:, :k, :3])

                with torch.no_grad():
                    z_t = encode_fn(x_t, bfloat16=bfloat16)
                    z_tpk = encode_fn(x_tpk, bfloat16=bfloat16)

                spatial_shuffle = bool(probe_cfg.get("spatial_shuffle", False))
                if spatial_shuffle:
                    z_hat = probe_ddp(z_t, a_t, spatial_shuffle=True)
                else:
                    z_hat = probe_ddp(z_t, a_t)

                if forward_loss == "huber":
                    loss = F.smooth_l1_loss(z_hat, z_tpk)
                else:
                    loss = F.mse_loss(z_hat, z_tpk)

                with torch.no_grad():
                    t = z_tpk.reshape(-1)
                    p = z_hat.reshape(-1)
                    ss_tot = torch.sum((t - t.mean()) ** 2)
                    ss_res = torch.sum((p - t) ** 2)
                    r2_z = 1.0 - ss_res / (ss_tot + 1e-9)
                    nmse_z = ss_res / (ss_tot + 1e-9)

                loss_logs = {
                    "loss/total": float(loss.detach().cpu()),
                    "train/r2_z": float(r2_z.detach().cpu()),
                    "train/nmse_z": float(nmse_z.detach().cpu()),
                }

            else:
                x, y, rel_t = batch[:3]
                x = x.to(device, non_blocking=True)     # (B,T,3,H,W)
                y = y.to(device, non_blocking=True)     # (B,G,3)
                rel_t = rel_t.to(device, non_blocking=True)

            if task not in {"forward", "forward_dynamics", "latent_forward", "latent_fwd", "fwd"}:
                B, T = x.shape[:2]
                G = y.shape[1]
                ctx = int(config["context_size"])
                assert T == ctx + G

            if task not in {"forward", "forward_dynamics", "latent_forward", "latent_fwd", "fwd"}:
                curr = x[:, ctx - 1]
                goal = x[:, ctx:]

                curr = curr.unsqueeze(1).expand(B, G, *curr.shape[1:]).reshape(B * G, *curr.shape[1:])
                goal = goal.reshape(B * G, *goal.shape[2:])
                tgt = y.reshape(B * G, 3)
                rel_flat = rel_t.reshape(B * G)

            if task not in {"forward", "forward_dynamics", "latent_forward", "latent_fwd", "fwd"}:
                if avoid_zero:
                    m = rel_flat.abs() > 1e-12
                    if m.sum() == 0:
                        continue
                    curr, goal, tgt, rel_flat = curr[m], goal[m], tgt[m], rel_flat[m]

            if task not in {"forward", "forward_dynamics", "latent_forward", "latent_fwd", "fwd"}:
                with torch.no_grad():
                    curr_lat = encode_fn(curr, bfloat16=bfloat16)
                    goal_lat = encode_fn(goal, bfloat16=bfloat16)

                    c_mean, c_std = curr_lat.mean().item(), curr_lat.std().item()
                    g_mean, g_std = goal_lat.mean().item(), goal_lat.std().item()

                    goal_lat = apply_latent_aug(goal_lat, sigma_min, sigma_max, tok_drop)

                pred = probe_ddp(curr_lat, goal_lat, rel_t=rel_flat)

            if task not in {"forward", "forward_dynamics", "latent_forward", "latent_fwd", "fwd"}:
                dxdy_p_n = pred[:, :2]
                dxdy_t_n = tgt[:, :2]
                if loss_cfg.xy_loss == "mse":
                    loss_xy = F.mse_loss(dxdy_p_n, dxdy_t_n)
                else:
                    loss_xy = F.smooth_l1_loss(dxdy_p_n, dxdy_t_n)

            if task not in {"forward", "forward_dynamics", "latent_forward", "latent_fwd", "fwd"}:
                dxdy_p = unnormalize_data(dxdy_p_n.detach(), action_stats)
                dxdy_t = unnormalize_data(dxdy_t_n, action_stats)

                err_dxdy = dxdy_p - dxdy_t
                mae_xy_step = torch.norm(err_dxdy, dim=-1).mean()
                mae_dx_step = err_dxdy[:, 0].abs().mean()
                mae_dy_step = err_dxdy[:, 1].abs().mean()

                std_dx_step = dxdy_t[:, 0].std(unbiased=False)
                std_dy_step = dxdy_t[:, 1].std(unbiased=False)

                r2_dx_step = r2_score(dxdy_p[:, 0], dxdy_t[:, 0])
                r2_dy_step = r2_score(dxdy_p[:, 1], dxdy_t[:, 1])

                yaw_t = tgt[:, 2]
                if loss_cfg.yaw_mode == "sincos":
                    sc_p = F.normalize(pred[:, 2:4], dim=-1)
                    sc_t = torch.stack([torch.sin(yaw_t), torch.cos(yaw_t)], dim=-1)
                    loss_yaw = F.mse_loss(sc_p, sc_t)
                else:
                    yaw_p = pred[:, 2]
                    dyaw = wrap_to_pi(yaw_p - yaw_t)
                    loss_yaw = F.smooth_l1_loss(dyaw, torch.zeros_like(dyaw))

                yaw_sim = yaw_similarity(pred, yaw_t, yaw_mode=str(loss_cfg.yaw_mode))

                loss = loss_cfg.xy_weight * loss_xy + loss_cfg.yaw_weight * loss_yaw
                loss_logs = {
                    "loss/xy": float(loss_xy.detach().cpu()),
                    "loss/yaw": float(loss_yaw.detach().cpu()),
                    "loss/total": float(loss.detach().cpu()),
                    "train/mae_xy_step": float(mae_xy_step.detach().cpu()),
                    "train/mae_dx_step": float(mae_dx_step.detach().cpu()),
                    "train/mae_dy_step": float(mae_dy_step.detach().cpu()),
                    "train/std_dx_step": float(std_dx_step.detach().cpu()),
                    "train/std_dy_step": float(std_dy_step.detach().cpu()),
                    "train/r2_dx_step": float(r2_dx_step.detach().cpu()),
                    "train/r2_dy_step": float(r2_dy_step.detach().cpu()),
                    "train/yaw_sim": float(yaw_sim.detach().cpu()),
                    "debug/lat_mean": c_mean,
                    "debug/lat_std": c_std,
                }

            # lr schedule (warmup + cosine decay)
            base_lr = float(probe_cfg.get("lr", 1e-4))
            min_lr = float(probe_cfg.get("min_lr", 1e-6))
            
            if warmup_steps > 0 and global_step < warmup_steps:
                # Linear warmup: 0 -> base_lr
                curr_lr = base_lr * float(global_step + 1) / float(warmup_steps)
            else:
                # Cosine decay: base_lr -> min_lr
                progress = float(global_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                progress = min(1.0, max(0.0, progress))
                curr_lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

            for pg in opt.param_groups:
                pg["lr"] = curr_lr

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe_ddp.parameters(), grad_clip)
            opt.step()

            running_loss += float(loss.detach().cpu())
            running_n += 1
            global_step += 1

            if global_step % log_every == 0:
                avg = running_loss / max(1, running_n)
                # distributed mean
                t = torch.tensor(avg, device=device)
                all_reduce_sum(t)
                t = t / dist.get_world_size()
                if rank == 0:
                    logger.info(f"step={global_step} loss={float(t.cpu()):.6f} lr={opt.param_groups[0]['lr']:.2e} {loss_logs}")
                    if config.get("wandb", {}).get("enabled", False):
                        payload = {"train/loss": float(t.cpu()), "train/lr": opt.param_groups[0]["lr"], **loss_logs}
                        viz_n = int(probe_cfg.get("viz_num_samples", 4))
                        if viz_n > 0:
                            try:
                                panel = _make_probe_viz_panel(curr, goal, pred, tgt, action_stats, yaw_mode=str(probe_cfg.get("yaw_mode", "sincos")), max_items=viz_n)
                                payload["train/viz"] = wandb.Image(panel)
                            except Exception:
                                pass
                        wandb.log(payload, step=global_step)
                running_loss, running_n = 0.0, 0

            if test_loader is not None and (global_step % eval_every == 0):
                if task in {"forward", "forward_dynamics", "latent_forward", "latent_fwd", "fwd"}:
                    metrics = eval_forward_dynamics_probe(probe_ddp, encode_fn, test_loader, config, device, bfloat16=bfloat16)
                else:
                    metrics = eval_probe(probe_ddp, encode_fn, test_loader, action_stats, config, device, bfloat16=bfloat16)

                if rank == 0:
                    logger.info(f"[EVAL] step={global_step} {metrics}")
                    if config.get("wandb", {}).get("enabled", False):
                        wandb.log(metrics, step=global_step)

                    if best_metric in metrics:
                        curr = float(metrics[best_metric])
                        improved = (curr > best) if best_mode == "max" else (curr < best)
                        if improved:
                            best = curr
                            best_path = os.path.join(ckpt_dir, best_fname)
                            torch.save(
                                {
                                    "probe": probe_ddp.module.state_dict(),
                                    "opt": opt.state_dict(),
                                    "global_step": global_step,
                                    "best_metric": best_metric,
                                    "best_metric_value": best,
                                    "probe_task": task_key,
                                    "config": config,
                                },
                                best_path
                            )
                            logger.info(f"Saved {os.path.basename(best_path)} ({best_metric}={best:.6f})")

                probe_ddp.train()

            # periodic save
            if global_step % save_every == 0 and rank == 0:
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{global_step}.pth")
                torch.save(
                    {
                        "probe": probe_ddp.module.state_dict(),
                        "opt": opt.state_dict(),
                        "global_step": global_step,
                        "best_metric": best_metric,
                        "best_metric_value": best,
                        "config": config,
                    },
                    ckpt_path
                )
                logger.info(f"Saved {ckpt_path}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if test_loader is not None:
        best_path = os.path.join(ckpt_dir, best_fname)
        legacy_best_path = os.path.join(ckpt_dir, "best.pth")
        used_ckpt = None

        load_path = best_path if os.path.isfile(best_path) else (legacy_best_path if os.path.isfile(legacy_best_path) else None)
        if load_path is not None:
            ckpt = torch.load(load_path, map_location={"cuda:0": f"cuda:{gpu}"})
            probe_sd = ckpt.get("probe", {})
            ckpt_is_forward = ("A.weight" in probe_sd) or ("B.weight" in probe_sd) or (ckpt.get("probe_task") == "forward_dynamics")
            if ckpt_is_forward == is_forward:
                probe_ddp.module.load_state_dict(probe_sd, strict=True)
                used_ckpt = load_path

        only_spatial_random_test = bool(args.only_spatial_random_test)
        action_modes = _parse_csv_list(args.test_action_modes)
        if len(action_modes) > 0:
            bad = [m for m in action_modes if str(m).lower().strip() not in {"shuffle", "shuffled", "permute"}]
            if len(bad) > 0:
                raise ValueError(f"Unsupported --test-action-modes={args.test_action_modes}. Supported: shuffle")

        if task in {"forward", "forward_dynamics", "latent_forward", "latent_fwd", "fwd"}:
            if len(action_modes) > 0:
                for am in action_modes:
                    metrics = eval_forward_dynamics_probe(
                        probe_ddp,
                        encode_fn,
                        test_loader,
                        config,
                        device,
                        bfloat16=bfloat16,
                        max_batches_override=len(test_loader),
                        spatial_shuffle=only_spatial_random_test,
                        action_override=str(am),
                        action_random_mode=str(args.action_random_mode),
                    )
                    if rank == 0:
                        suffix = f"action_{str(am).lower().strip()}"
                        if only_spatial_random_test:
                            suffix = suffix + "_spatial_shuffle"
                        out_name = f"test_metrics_{suffix}.json"
                        out_path = os.path.join(exp_dir, out_name)
                        payload = {
                            "global_step": int(global_step),
                            "checkpoint": used_ckpt,
                            "action_override": str(am),
                            "spatial_shuffle": bool(only_spatial_random_test),
                            **metrics,
                        }
                        with open(out_path, "w") as f:
                            json.dump(payload, f, indent=2, sort_keys=True)
                        logger.info(f"Saved test metrics to {out_path}: {payload}")
            else:
                metrics = eval_forward_dynamics_probe(
                    probe_ddp,
                    encode_fn,
                    test_loader,
                    config,
                    device,
                    bfloat16=bfloat16,
                    max_batches_override=len(test_loader),
                    spatial_shuffle=only_spatial_random_test,
                )
                if rank == 0:
                    out_name = "test_metrics_random.json" if only_spatial_random_test else "test_metrics.json"
                    out_path = os.path.join(exp_dir, out_name)
                    payload = {"global_step": int(global_step), "checkpoint": used_ckpt, **metrics}
                    with open(out_path, "w") as f:
                        json.dump(payload, f, indent=2, sort_keys=True)
                    logger.info(f"Saved test metrics to {out_path}: {payload}")
        else:
            metrics = eval_probe(
                probe_ddp,
                encode_fn,
                test_loader,
                action_stats,
                config,
                device,
                bfloat16=bfloat16,
                max_batches_override=len(test_loader),
            )

            if rank == 0:
                out_name = "test_metrics_random.json" if only_spatial_random_test else "test_metrics.json"
                out_path = os.path.join(exp_dir, out_name)
                payload = {"global_step": int(global_step), "checkpoint": used_ckpt, **metrics}
                with open(out_path, "w") as f:
                    json.dump(payload, f, indent=2, sort_keys=True)
                logger.info(f"Saved test metrics to {out_path}: {payload}")

    if rank == 0:
        logger.info("Training finished.")
        if config.get("wandb", {}).get("enabled", False):
            wandb.finish()


if __name__ == "__main__":
    main()
