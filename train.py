"""
Train script for diffusion-based waypoint prediction model.
Adapted for NPZ data format with logits as input maps.
### CHANGED: Adapted for 2D (x,y) prediction only (removed yaw).
### CHANGED: Added RMSE and Inter-Waypoint Spacing metrics.
### CHANGED: Added Resume Training functionality.
"""

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from typing import Tuple, Optional, Dict, Any, List
import json
import csv
from datetime import datetime
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Local imports - use absolute imports
import model
import dataloader
build_model = model.build_model
NPZWaypointDataset = dataloader.NPZWaypointDataset

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Visualization imports
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available, plotting disabled")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available, using CSV logging only")


class TrainingLogger:
    """
    Custom training logger for recording training progress and metrics.
    Saves logs in CSV format and generates visualizations.
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Training log (records every batch)
        self.train_csv_path = os.path.join(log_dir, "training_log.csv")
        with open(self.train_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'epoch_step', 'global_step', 'batch_idx',
                'loss', 'lr', 'grad_norm', 'timestamp'
            ])
        
        # Validation log (records each validation step)
        self.val_csv_path = os.path.join(log_dir, "validation_log.csv")
        with open(self.val_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # CHANGED: Added 'rmse' and 'avg_spacing' to headers
            writer.writerow([
                'epoch', 'global_step', 'val_mse', 'ade', 'fde', 'rmse', 'avg_spacing',
                'best_metric', 'best_score', 'timestamp'
            ])
        
        # Epoch summary log (records each epoch's average metrics)
        self.epoch_csv_path = os.path.join(log_dir, "epoch_summary.csv")
        with open(self.epoch_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # CHANGED: Added 'rmse' and 'avg_spacing' to headers
            writer.writerow([
                'epoch', 'train_loss_avg', 'val_mse', 'ade', 'fde', 'rmse', 'avg_spacing',
                'lr', 'num_batches', 'timestamp'
            ])
        
        # Create directories for plots
        self.plot_dir = os.path.join(log_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Save configuration
        self.config_path = os.path.join(log_dir, "config.json")
    
    def log_training(self, epoch: int, epoch_step: int, global_step: int, 
                     batch_idx: int, loss: float, lr: float, grad_norm: float):
        """Log training metrics for a single batch."""
        with open(self.train_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, epoch_step, global_step, batch_idx,
                f"{loss:.6f}", f"{lr:.6f}", f"{grad_norm:.6f}",
                datetime.now().isoformat()
            ])
    
    # CHANGED: Added rmse and avg_spacing arguments
    def log_validation(self, epoch: int, global_step: int, val_mse: float,
                       ade: float, fde: float, rmse: float, avg_spacing: float,
                       best_metric: str, best_score: float):
        """Log validation metrics."""
        with open(self.val_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, global_step, f"{val_mse:.6f}", 
                f"{ade:.4f}", f"{fde:.4f}", f"{rmse:.4f}", f"{avg_spacing:.4f}",
                best_metric, f"{best_score:.6f}",
                datetime.now().isoformat()
            ])
    
    # CHANGED: Added rmse and avg_spacing arguments
    def log_epoch_summary(self, epoch: int, train_loss_avg: float, 
                          val_mse: float, ade: float, fde: float, 
                          rmse: float, avg_spacing: float,
                          lr: float, num_batches: int):
        """Log summary metrics for an entire epoch."""
        with open(self.epoch_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{train_loss_avg:.6f}", f"{val_mse:.6f}",
                f"{ade:.4f}", f"{fde:.4f}", f"{rmse:.4f}", f"{avg_spacing:.4f}",
                f"{lr:.6f}", num_batches, datetime.now().isoformat()
            ])
    
    def save_config(self, config: dict):
        """Save training configuration as JSON."""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def plot_loss_curves(self, downsample_factor: int = 10, 
                         smooth_window: int = 100):
        """Generate loss curves from logged data."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plot generation")
            return
        
        try:
            # Load training data
            train_data = []
            with open(self.train_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i % downsample_factor == 0:
                        train_data.append({
                            'global_step': int(row['global_step']),
                            'loss': float(row['loss'])
                        })
            
            # Load validation data
            val_data = []
            with open(self.val_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    val_data.append({
                        'global_step': int(row['global_step']),
                        'val_mse': float(row['val_mse']),
                        'ade': float(row['ade']),
                        'fde': float(row['fde']),
                        'rmse': float(row['rmse']),         # CHANGED: Read new metric
                        'avg_spacing': float(row['avg_spacing']) # CHANGED: Read new metric
                    })
            
            # Load epoch summary data
            epoch_data = []
            with open(self.epoch_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epoch_data.append({
                        'epoch': int(row['epoch']),
                        'train_loss_avg': float(row['train_loss_avg']),
                        'val_mse': float(row['val_mse']),
                        'ade': float(row['ade']),
                        'fde': float(row['fde']),
                        'rmse': float(row['rmse']),
                        'avg_spacing': float(row['avg_spacing'])
                    })
            
            # Create comprehensive visualization
            fig = plt.figure(figsize=(18, 12))
            
            # Plot 1: Training loss (global step)
            ax1 = plt.subplot(2, 3, 1)
            if train_data:
                steps = [d['global_step'] for d in train_data]
                losses = [d['loss'] for d in train_data]
                ax1.plot(steps, losses, 'b-', alpha=0.3, linewidth=0.5, label='Raw')
                if len(losses) > smooth_window:
                    smooth_losses = []
                    smooth_steps = []
                    for i in range(0, len(losses), smooth_window):
                        smooth_steps.append(steps[i])
                        smooth_losses.append(losses[i])
                    ax1.plot(smooth_steps, smooth_losses, 'b-', linewidth=2, label=f'Smoothed')
                ax1.set_xlabel('Global Step')
                ax1.set_ylabel('Training Loss')
                ax1.set_title('Training Loss Curve')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Plot 2: Validation MSE (global step)
            ax2 = plt.subplot(2, 3, 2)
            if val_data:
                steps = [d['global_step'] for d in val_data]
                val_losses = [d['val_mse'] for d in val_data]
                ax2.plot(steps, val_losses, 'r-', linewidth=2, marker='o', markersize=6)
                ax2.set_xlabel('Global Step')
                ax2.set_ylabel('Validation MSE')
                ax2.set_title('Validation MSE (Noise Prediction)')
                ax2.grid(True, alpha=0.3)
            
            # Plot 3: ADE, FDE, RMSE (global step)
            ax3 = plt.subplot(2, 3, 3)
            if val_data:
                steps = [d['global_step'] for d in val_data]
                ade_vals = [d['ade'] for d in val_data]
                rmse_vals = [d['rmse'] for d in val_data]
                ax3.plot(steps, ade_vals, 'g-', label='ADE', linewidth=2, marker='s')
                ax3.plot(steps, rmse_vals, 'b--', label='RMSE', linewidth=2, marker='x')
                ax3.set_xlabel('Global Step')
                ax3.set_ylabel('Error (meters)')
                ax3.set_title('Path Reconstruction Errors')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Average Spacing (global step) - CHANGED: New Plot
            ax4 = plt.subplot(2, 3, 4)
            if val_data:
                steps = [d['global_step'] for d in val_data]
                spacing_vals = [d['avg_spacing'] for d in val_data]
                ax4.plot(steps, spacing_vals, 'c-', linewidth=2, marker='d')
                ax4.axhline(y=0.10, color='k', linestyle='--', label='Target (0.1m)')
                ax4.set_xlabel('Global Step')
                ax4.set_ylabel('Spacing (m)')
                ax4.set_title('Avg Inter-Waypoint Spacing')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            # Plot 5: Training vs Validation loss (epoch)
            ax5 = plt.subplot(2, 3, 5)
            if epoch_data and len(epoch_data) > 1:
                epochs = [d['epoch'] for d in epoch_data]
                train_losses = [d['train_loss_avg'] for d in epoch_data]
                val_losses = [d['val_mse'] for d in epoch_data]
                ax5.plot(epochs, train_losses, 'b-', label='Train Loss')
                ax5.plot(epochs, val_losses, 'r-', label='Val MSE')
                ax5.set_xlabel('Epoch')
                ax5.set_title('Loss per Epoch')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
            
            # Plot 6: Metric Summary (epoch)
            ax6 = plt.subplot(2, 3, 6)
            if epoch_data and len(epoch_data) > 1:
                epochs = [d['epoch'] for d in epoch_data]
                ade_vals = [d['ade'] for d in epoch_data]
                rmse_vals = [d['rmse'] for d in epoch_data]
                ax6.plot(epochs, ade_vals, 'g-', label='ADE')
                ax6.plot(epochs, rmse_vals, 'b--', label='RMSE')
                ax6.set_xlabel('Epoch')
                ax6.set_title('Metrics per Epoch')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(self.plot_dir, "comprehensive_training_plots.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Training plots saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            import traceback
            traceback.print_exc()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    """Training configuration dataclass."""
    
    # Data configuration
    train_dir: str
    val_dir: Optional[str]
    # CHANGED: Added resume_path
    resume_path: Optional[str] = None
    
    image_height: int = 1616
    image_width: int = 1232
    patch_size: int = 8
    num_waypoints: int = 20
    xy_scale: float = 1.0
    
    # Model configuration
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.0
    use_learned_img_pos: bool = False
    
    # Model modes and conditioning
    mode: str = 'encoder_decoder'
    causal_attn: bool = False
    time_as_cond: bool = True
    obs_as_cond: bool = False
    cond_dim: int = 0
    n_cond_layers: int = 0
    vision_n_layers: int = 0
    
    # Diffusion configuration
    num_diffusion_steps: int = 1000
    beta_schedule: str = 'squaredcos_cap_v2'
    
    # Optimization configuration
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    
    # Training configuration
    seed: int = 42
    device: str = 'cuda'
    amp: bool = True
    log_interval: int = 10
    out_dir: str = './outputs'
    
    # Evaluation and logging configuration
    eval_every_steps: int = 500
    eval_num_inference_steps: int = 50
    max_eval_batches: int = 10
    save_checkpoint_every: int = 5
    
    # Model selection
    # CHANGED: Added 'rmse' and 'avg_spacing' to choices
    best_metric: str = 'ade'  # Options: 'ade', 'fde', 'noise_mse', 'rmse', 'avg_spacing'
    
    # Logging and visualization
    use_tensorboard: bool = True
    save_plots: bool = True
    verbose_logging: bool = True


def create_dataloaders(cfg: TrainConfig):
    """Create training and validation data loaders."""
    logger.info(f"Creating dataloaders with:")
    logger.info(f"  Image size: {cfg.image_height}x{cfg.image_width}")
    logger.info(f"  Num waypoints: {cfg.num_waypoints}")
    logger.info(f"  Batch size: {cfg.batch_size}")
    
    train_set = NPZWaypointDataset(
        cfg.train_dir,
        image_size=(cfg.image_height, cfg.image_width),
        num_waypoints=cfg.num_waypoints,
        patch_size=cfg.patch_size,
        xy_scale=cfg.xy_scale,
        verbose=cfg.verbose_logging
    )
    
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = None
    if cfg.val_dir is not None and os.path.isdir(cfg.val_dir):
        val_set = NPZWaypointDataset(
            cfg.val_dir,
            image_size=(cfg.image_height, cfg.image_width),
            num_waypoints=cfg.num_waypoints,
            patch_size=cfg.patch_size,
            xy_scale=cfg.xy_scale,
            verbose=cfg.verbose_logging
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=min(cfg.batch_size, 4),
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
        logger.info(f"Validation set: {len(val_set)} samples")
    
    logger.info(f"Training set: {len(train_set)} samples")
    logger.info(f"Number of batches per epoch: {len(train_loader)}")
    
    return train_loader, val_loader


def evaluate_noise_prediction(model: nn.Module, data_loader: DataLoader, 
                              scheduler: DDPMScheduler, device: torch.device) -> float:
    """Evaluate model on validation set for noise prediction MSE."""
    model.eval()
    mse_sum = 0.0
    count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            img = batch['map'].to(device)
            wp = batch['waypoints'].to(device)
            goal = batch['goal'].to(device)
            
            B = img.shape[0]
            t = torch.randint(
                0, scheduler.config.num_train_timesteps, 
                (B,), device=device
            ).long()
            
            noise = torch.randn_like(wp)
            noisy = scheduler.add_noise(original_samples=wp, noise=noise, timesteps=t)
            
            pred = model(img, noisy, t, goal=goal)
            loss = F.mse_loss(pred, noise, reduction='sum')
            
            mse_sum += loss.item()
            count += wp.numel()
            
            if batch_idx % 10 == 0 and logger.isEnabledFor(logging.INFO):
                logger.info(f"  Validation batch {batch_idx}/{len(data_loader)}")
    
    model.train()
    return mse_sum / max(1, count)


@torch.no_grad()
def evaluate_sampling_performance(
    model: nn.Module,
    data_loader: DataLoader,
    scheduler: DDPMScheduler,
    device: torch.device,
    xy_scale: float,
    num_inference_steps: int = 50,
    max_batches: int = 10,
    verbose: bool = True
) -> Dict[str, float]:
    """Evaluate sampling performance with new metrics (RMSE, Spacing)."""
    model.eval()
    
    scheduler.set_timesteps(num_inference_steps)
    
    ade_sum = 0.0
    fde_sum = 0.0
    rmse_sum = 0.0        # CHANGED: Accumulator for RMSE
    spacing_sum = 0.0     # CHANGED: Accumulator for Spacing
    n_count = 0
    batches_done = 0
    
    for batch_idx, batch in enumerate(data_loader):
        img = batch['map'].to(device)
        gt = batch['waypoints'].to(device)
        goal = batch['goal'].to(device)
        
        B, K, _ = gt.shape
        
        # Initialize from Gaussian noise
        sample = torch.randn_like(gt)
        
        # Pre-encode vision and goal tokens (reused across all steps)
        vision_tokens = model._forward_vision(img)
        goal_tokens = model._forward_goal(goal)
        
        # Reverse diffusion process
        for step_idx, t in enumerate(scheduler.timesteps):
            eps = model._denoise(sample, t, vision_tokens, goal_tokens)
            sample = scheduler.step(model_output=eps, timestep=t, sample=sample).prev_sample
            
            if verbose and batch_idx == 0 and step_idx % 10 == 0:
                logger.info(f"    Denoising step {step_idx}/{len(scheduler.timesteps)}")
        
        # Denormalize predictions
        pred_xy = sample[..., :2] * xy_scale
        gt_xy = gt[..., :2] * xy_scale
        
        # 1. Standard Metrics (ADE / FDE)
        pos_err = torch.linalg.vector_norm(pred_xy - gt_xy, dim=-1)  # [B, K]
        ade = pos_err.mean(dim=1)  # [B]
        fde = pos_err[:, -1]  # [B]
        
        # 2. New Metric: RMSE (Root Mean Squared Error)
        # Calculate squared error for all points, mean, then sqrt.
        mse_per_sample = ((pred_xy - gt_xy) ** 2).sum(dim=-1).mean(dim=-1) # [B]
        rmse = torch.sqrt(mse_per_sample) # [B]
        
        # 3. New Metric: Average Inter-Waypoint Spacing
        # Calculate distance between P_i and P_{i+1}
        diffs = pred_xy[:, 1:, :] - pred_xy[:, :-1, :]
        dists = torch.linalg.vector_norm(diffs, dim=-1) # [B, K-1]
        avg_dist = dists.mean(dim=-1) # [B]
        
        ade_sum += ade.sum().item()
        fde_sum += fde.sum().item()
        rmse_sum += rmse.sum().item()           # CHANGED: Accumulate RMSE
        spacing_sum += avg_dist.sum().item()    # CHANGED: Accumulate Spacing
        
        n_count += B
        batches_done += 1
        
        if verbose:
            logger.info(f"  Sampling batch {batches_done}: "
                       f"ADE={ade.mean().item():.3f}, "
                       f"RMSE={rmse.mean().item():.3f}, "
                       f"Space={avg_dist.mean().item():.3f}m")
        
        if batches_done >= max_batches:
            break
    
    metrics = {
        'ade': ade_sum / max(1, n_count),
        'fde': fde_sum / max(1, n_count),
        'rmse': rmse_sum / max(1, n_count),          # CHANGED: Return RMSE
        'avg_spacing': spacing_sum / max(1, n_count) # CHANGED: Return Spacing
    }
    
    model.train()
    return metrics


def train(cfg: TrainConfig):
    """Main training function."""
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.out_dir = os.path.join(cfg.out_dir, f"train_{timestamp}")
    os.makedirs(cfg.out_dir, exist_ok=True)
    
    # Set random seed
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    
    # Print configuration
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Output directory: {cfg.out_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Image size: {cfg.image_height}x{cfg.image_width}")
    logger.info(f"Num waypoints: {cfg.num_waypoints}")
    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"Epochs: {cfg.epochs}")
    logger.info("=" * 80)
    
    # Initialize logger
    training_logger = TrainingLogger(cfg.out_dir)
    training_logger.save_config(asdict(cfg))
    
    # Build model
    logger.info("Building model...")
    model_obj = build_model(
        image_size=(cfg.image_height, cfg.image_width),
        patch_size=cfg.patch_size,
        in_chans=1,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        num_waypoints=cfg.num_waypoints,
        dropout=cfg.dropout,
        attn_dropout=cfg.attn_dropout,
        use_learned_img_pos=cfg.use_learned_img_pos,
        mode=cfg.mode,
        causal_attn=cfg.causal_attn,
        time_as_cond=cfg.time_as_cond,
        obs_as_cond=cfg.obs_as_cond,
        cond_dim=cfg.cond_dim,
        n_cond_layers=cfg.n_cond_layers,
        vision_n_layers=cfg.vision_n_layers,
    ).to(device)
    
    # CHANGED: Added Resume Training Logic
    # If a resume path is provided, load weights before optimizer initialization
    if cfg.resume_path and os.path.exists(cfg.resume_path):
        logger.info(f"============================================================")
        logger.info(f"RESUMING TRAINING FROM: {cfg.resume_path}")
        logger.info(f"============================================================")
        try:
            ckpt = torch.load(cfg.resume_path, map_location=device)
            # Load model weights
            if 'model_state_dict' in ckpt:
                model_obj.load_state_dict(ckpt['model_state_dict'])
            else:
                # Fallback if checkpoint is just the state dict
                model_obj.load_state_dict(ckpt)
            logger.info("Weights loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise e
    
    # Print model statistics
    total_params = sum(p.numel() for p in model_obj.parameters())
    trainable_params = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
    logger.info(f"Model built. Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Diffusion scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_diffusion_steps,
        beta_schedule=cfg.beta_schedule,
        clip_sample=True,
        prediction_type='epsilon',
    )
    
    # Data loaders
    train_loader, val_loader = create_dataloaders(cfg)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model_obj.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == 'cuda')
    
    # TensorBoard writer
    writer = None
    if cfg.use_tensorboard and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=os.path.join(cfg.out_dir, 'tensorboard'))
        logger.info(f"TensorBoard logs at: {os.path.join(cfg.out_dir, 'tensorboard')}")
    
    # Training state
    best_score = float('inf')
    best_ckpt_path = os.path.join(cfg.out_dir, 'best_model.pt')
    global_step = 0
    
    logger.info("\nStarting training...")
    
    # Training loop
    for epoch in range(cfg.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{cfg.epochs}")
        logger.info(f"{'='*60}")
        
        # Training phase
        model_obj.train()
        epoch_loss_sum = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            img = batch['map'].to(device)
            wp = batch['waypoints'].to(device)
            goal = batch['goal'].to(device)
            
            B = img.shape[0]
            
            # Sample random timesteps
            t = torch.randint(
                0, scheduler.config.num_train_timesteps, 
                (B,), device=device
            ).long()
            
            # Add noise to waypoints
            noise = torch.randn_like(wp)
            noisy = scheduler.add_noise(original_samples=wp, noise=noise, timesteps=t)
            
            # Forward pass with mixed precision
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == 'cuda'):
                pred = model_obj(img, noisy, t, goal=goal)
                # REMINDER: This is the ONLY loss that propagates backward
                loss = F.mse_loss(pred, noise)
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_obj.parameters(), cfg.grad_clip_norm)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Log training metrics
            loss_value = loss.item()
            epoch_loss_sum += loss_value
            batch_count += 1
            
            # Calculate gradient norm
            total_norm = 0.0
            for p in model_obj.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # Record training step
            current_lr = optimizer.param_groups[0]['lr']
            training_logger.log_training(
                epoch=epoch+1,
                epoch_step=batch_idx+1,
                global_step=global_step,
                batch_idx=batch_idx,
                loss=loss_value,
                lr=current_lr,
                grad_norm=total_norm
            )
            
            # TensorBoard logging
            if writer is not None:
                writer.add_scalar('train/loss', loss_value, global_step)
                writer.add_scalar('train/lr', current_lr, global_step)
                writer.add_scalar('train/grad_norm', total_norm, global_step)
            
            # Print progress
            if (batch_idx + 1) % cfg.log_interval == 0:
                avg_loss = epoch_loss_sum / batch_count
                logger.info(f"  Batch {batch_idx+1}/{len(train_loader)}: "
                          f"Loss = {loss_value:.6f}, Avg = {avg_loss:.6f}, "
                          f"LR = {current_lr:.6f}")
            
            global_step += 1
            
            # Periodic validation
            if cfg.eval_every_steps and (global_step % cfg.eval_every_steps == 0):
                if val_loader is not None:
                    logger.info(f"\n  Step {global_step}: Running validation...")
                    
                    # Evaluate noise prediction
                    val_mse = evaluate_noise_prediction(model_obj, val_loader, scheduler, device)
                    logger.info(f"  Validation MSE = {val_mse:.6f}")
                    
                    # Evaluate sampling performance
                    logger.info("  Running sampling evaluation...")
                    samp_metrics = evaluate_sampling_performance(
                        model_obj, val_loader, scheduler, device,
                        xy_scale=cfg.xy_scale, 
                        num_inference_steps=cfg.eval_num_inference_steps,
                        max_batches=cfg.max_eval_batches,
                        verbose=cfg.verbose_logging
                    )
                    
                    # CHANGED: Added logging for RMSE and Spacing
                    logger.info(f"  ADE = {samp_metrics['ade']:.4f}, "
                              f"RMSE = {samp_metrics['rmse']:.4f}, "
                              f"Space = {samp_metrics['avg_spacing']:.4f}m")
                    
                    # Record validation metrics
                    training_logger.log_validation(
                        epoch=epoch+1,
                        global_step=global_step,
                        val_mse=val_mse,
                        ade=samp_metrics['ade'],
                        fde=samp_metrics['fde'],
                        rmse=samp_metrics['rmse'],             # CHANGED: Log RMSE
                        avg_spacing=samp_metrics['avg_spacing'], # CHANGED: Log Spacing
                        best_metric=cfg.best_metric,
                        best_score=best_score
                    )
                    
                    # TensorBoard logging
                    if writer is not None:
                        writer.add_scalar('val/mse', val_mse, global_step)
                        writer.add_scalar('val/ade', samp_metrics['ade'], global_step)
                        writer.add_scalar('val/fde', samp_metrics['fde'], global_step)
                        writer.add_scalar('val/rmse', samp_metrics['rmse'], global_step)       # CHANGED
                        writer.add_scalar('val/spacing', samp_metrics['avg_spacing'], global_step) # CHANGED
                    
                    # Update learning rate scheduler
                    lr_scheduler.step(val_mse)
                    
                    # Save best model checkpoint
                    metric_map = {
                        'ade': samp_metrics['ade'],
                        'fde': samp_metrics['fde'],
                        'noise_mse': val_mse,
                        'rmse': samp_metrics['rmse'],      # CHANGED
                        'avg_spacing': samp_metrics['avg_spacing'], # CHANGED
                    }
                    
                    current_score = metric_map.get(cfg.best_metric)
                    if current_score is None:
                        raise ValueError(f"Unknown best_metric {cfg.best_metric}")
                    
                    if current_score < best_score:
                        best_score = current_score
                        checkpoint = {
                            'model_state_dict': model_obj.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'config': asdict(cfg),
                            'metrics': {
                                'noise_mse': val_mse,
                                **samp_metrics,
                            },
                            'best_metric': cfg.best_metric,
                            'best_score': float(best_score),
                            'global_step': global_step,
                            'epoch': epoch + 1,
                        }
                        torch.save(checkpoint, best_ckpt_path)
                        logger.info(f"  [Step] Saved new best checkpoint ({cfg.best_metric}={best_score:.6f})")
        
        # End of epoch validation
        epoch_avg_loss = epoch_loss_sum / max(1, batch_count)
        logger.info(f"\nEpoch {epoch+1} completed:")
        logger.info(f"  Average training loss: {epoch_avg_loss:.6f}")
        
        if val_loader is not None:
            logger.info("  Running end-of-epoch validation...")
            
            val_mse = evaluate_noise_prediction(model_obj, val_loader, scheduler, device)
            logger.info(f"  Validation MSE: {val_mse:.6f}")
            
            # Sampling evaluation
            samp_metrics = evaluate_sampling_performance(
                model_obj, val_loader, scheduler, device,
                xy_scale=cfg.xy_scale, 
                num_inference_steps=cfg.eval_num_inference_steps,
                max_batches=cfg.max_eval_batches,
                verbose=cfg.verbose_logging
            )
            
            # CHANGED: Log RMSE and Spacing
            logger.info(f"  ADE: {samp_metrics['ade']:.4f}, "
                      f"RMSE: {samp_metrics['rmse']:.4f}, "
                      f"Space: {samp_metrics['avg_spacing']:.4f}")
            
            # Log epoch summary
            current_lr = optimizer.param_groups[0]['lr']
            training_logger.log_epoch_summary(
                epoch=epoch+1,
                train_loss_avg=epoch_avg_loss,
                val_mse=val_mse,
                ade=samp_metrics['ade'],
                fde=samp_metrics['fde'],
                rmse=samp_metrics['rmse'],             # CHANGED
                avg_spacing=samp_metrics['avg_spacing'], # CHANGED
                lr=current_lr,
                num_batches=batch_count
            )
            
            # Save periodic checkpoint
            # This logic ensures models are saved every 5 epochs (or whatever cfg.save_checkpoint_every is)
            if (epoch + 1) % cfg.save_checkpoint_every == 0:
                ckpt_path = os.path.join(cfg.out_dir, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'model_state_dict': model_obj.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': asdict(cfg),
                    'epoch': epoch + 1,
                    'global_step': global_step,
                }, ckpt_path)
                logger.info(f"  Saved checkpoint to {ckpt_path}")
        
        # Save latest checkpoint
        last_ckpt_path = os.path.join(cfg.out_dir, 'latest_model.pt')
        torch.save({
            'model_state_dict': model_obj.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': asdict(cfg),
            'epoch': epoch + 1,
            'global_step': global_step,
        }, last_ckpt_path)
    
    # Training completed
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING COMPLETED")
    logger.info(f"{'='*80}")
    
    # Generate training plots
    if cfg.save_plots:
        training_logger.plot_loss_curves(downsample_factor=10, smooth_window=50)
    
    # Save final model
    final_ckpt_path = os.path.join(cfg.out_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model_obj.state_dict(),
        'config': asdict(cfg),
        'final_metrics': {
            'best_score': best_score,
            'best_metric': cfg.best_metric,
        }
    }, final_ckpt_path)
    logger.info(f"Final model saved to {final_ckpt_path}")
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    # Print summary
    logger.info("\nTraining Summary:")
    logger.info(f"  Best {cfg.best_metric}: {best_score:.6f}")
    logger.info(f"  Total training steps: {global_step}")
    logger.info(f"  All logs saved to: {cfg.out_dir}")
    logger.info(f"{'='*80}")


def build_argparser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Train diffusion waypoint predictor on NPZ dataset'
    )
    
    # Data arguments
    parser.add_argument('--train-dir', type=str, required=True,
                       help='Path to directory of .npz training files')
    parser.add_argument('--val-dir', type=str, default=None,
                       help='Optional path to directory of .npz validation files')
    # CHANGED: Added resume arg
    parser.add_argument('--resume-path', type=str, default=None,
                       help='Path to checkpoint file (.pt) to resume training from')
                       
    parser.add_argument('--image-height', type=int, default=1616,
                       help='Image height (your logits height)')
    parser.add_argument('--image-width', type=int, default=1232,
                       help='Image width (your logits width)')
    parser.add_argument('--patch-size', type=int, default=8)
    parser.add_argument('--num-waypoints', type=int, default=20,
                       help='Number of waypoints (your data has 20)')
    parser.add_argument('--xy-scale', type=float, default=1.0,
                       help='Normalize x,y by this scale')
    
    # Model arguments
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--mlp-ratio', type=float, default=4.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attn-dropout', type=float, default=0.0)
    parser.add_argument('--use-learned-img-pos', action='store_true')
    
    # Model modes / conditioning
    parser.add_argument('--mode', type=str, default='encoder_decoder',
                       choices=['encoder', 'decoder', 'encoder_decoder'])
    parser.add_argument('--causal-attn', action='store_true')
    parser.add_argument('--no-time-as-cond', action='store_true',
                       help='If set, add time to target tokens instead of as cond token')
    parser.add_argument('--obs-as-cond', action='store_true',
                       help='Use extra observation conditioning tokens (requires cond_dim>0)')
    parser.add_argument('--cond-dim', type=int, default=0,
                       help='Dimensionality of obs cond tokens if used')
    parser.add_argument('--n-cond-layers', type=int, default=0,
                       help='Number of transformer layers for cond encoder (0 uses MLP)')
    parser.add_argument('--vision-n-layers', type=int, default=0,
                       help='Optional transformer encoder layers over vision tokens')
    
    # Diffusion arguments
    parser.add_argument('--num-diffusion-steps', type=int, default=1000)
    parser.add_argument('--beta-schedule', type=str, default='squaredcos_cap_v2')
    
    # Optimization arguments
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Reduced for large input size')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip_norm', type=float, default=1.0)
    
    # Training arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Log more frequently')
    parser.add_argument('--out-dir', type=str, default='./outputs')
    parser.add_argument('--eval-every-steps', type=int, default=500)
    parser.add_argument('--eval-num-inference-steps', type=int, default=50)
    parser.add_argument('--max-eval-batches', type=int, default=10)
    parser.add_argument('--save-checkpoint-every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    # CHANGED: Added 'rmse' and 'avg_spacing'
    parser.add_argument('--best-metric', type=str, default='ade',
                       choices=['ade', 'fde', 'noise_mse', 'rmse', 'avg_spacing'])
    
    # Logging and visualization arguments
    parser.add_argument('--no-tensorboard', action='store_true',
                       help='Disable TensorBoard logging')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable loss curve plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    
    return parser


def _load_yaml_config(path: str) -> dict:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config or {}


def _merge_configurations(args: argparse.Namespace) -> TrainConfig:
    """Merge command line arguments and YAML configuration."""
    if args.config is None:
        # Build configuration from command line only
        return TrainConfig(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            # CHANGED: Map resume path
            resume_path=args.resume_path,
            
            image_height=args.image_height,
            image_width=args.image_width,
            patch_size=args.patch_size,
            num_waypoints=args.num_waypoints,
            xy_scale=args.xy_scale,
            
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            attn_dropout=args.attn_dropout,
            use_learned_img_pos=args.use_learned_img_pos,
            
            mode=args.mode,
            causal_attn=args.causal_attn,
            time_as_cond=not args.no_time_as_cond,
            obs_as_cond=args.obs_as_cond,
            cond_dim=args.cond_dim,
            n_cond_layers=args.n_cond_layers,
            vision_n_layers=args.vision_n_layers,
            
            num_diffusion_steps=args.num_diffusion_steps,
            beta_schedule=args.beta_schedule,
            
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            
            seed=args.seed,
            device=args.device,
            amp=not args.no_amp,
            log_interval=args.log_interval,
            out_dir=args.out_dir,
            
            eval_every_steps=args.eval_every_steps,
            eval_num_inference_steps=args.eval_num_inference_steps,
            max_eval_batches=args.max_eval_batches,
            save_checkpoint_every=args.save_checkpoint_every,
            best_metric=args.best_metric,
            
            use_tensorboard=not args.no_tensorboard,
            save_plots=not args.no_plots,
            verbose_logging=args.verbose,
        )
    
    # Load YAML and override with command line arguments
    yaml_config = _load_yaml_config(args.config)
    
    def get_config_value(keys, default):
        """Helper to get nested configuration values."""
        value = yaml_config
        for key in keys:
            if value is None or key not in value:
                return default
            value = value[key]
        return value
    
    return TrainConfig(
        # Data
        train_dir=get_config_value(['data', 'train_dir'], args.train_dir),
        val_dir=get_config_value(['data', 'val_dir'], args.val_dir),
        resume_path=get_config_value(['data', 'resume_path'], args.resume_path),
        image_height=get_config_value(['model', 'image_height'], args.image_height),
        image_width=get_config_value(['model', 'image_width'], args.image_width),
        patch_size=get_config_value(['model', 'patch_size'], args.patch_size),
        num_waypoints=get_config_value(['model', 'num_waypoints'], args.num_waypoints),
        xy_scale=get_config_value(['data', 'xy_scale'], args.xy_scale),
        
        # Model
        embed_dim=get_config_value(['model', 'embed_dim'], args.embed_dim),
        depth=get_config_value(['model', 'depth'], args.depth),
        num_heads=get_config_value(['model', 'num_heads'], args.num_heads),
        mlp_ratio=get_config_value(['model', 'mlp_ratio'], args.mlp_ratio),
        dropout=get_config_value(['model', 'dropout'], args.dropout),
        attn_dropout=get_config_value(['model', 'attn_dropout'], args.attn_dropout),
        use_learned_img_pos=get_config_value(['model', 'use_learned_img_pos'], args.use_learned_img_pos),
        
        # Modes
        mode=get_config_value(['model', 'mode'], args.mode),
        causal_attn=get_config_value(['model', 'causal_attn'], args.causal_attn),
        time_as_cond=get_config_value(['model', 'time_as_cond'], not args.no_time_as_cond),
        obs_as_cond=get_config_value(['model', 'obs_as_cond'], args.obs_as_cond),
        cond_dim=get_config_value(['model', 'cond_dim'], args.cond_dim),
        n_cond_layers=get_config_value(['model', 'n_cond_layers'], args.n_cond_layers),
        vision_n_layers=get_config_value(['model', 'vision_n_layers'], args.vision_n_layers),
        
        # Diffusion
        num_diffusion_steps=get_config_value(['diffusion', 'num_train_timesteps'], args.num_diffusion_steps),
        beta_schedule=get_config_value(['diffusion', 'beta_schedule'], args.beta_schedule),
        
        # Optimization
        epochs=get_config_value(['train', 'epochs'], args.epochs),
        batch_size=get_config_value(['train', 'batch_size'], args.batch_size),
        lr=get_config_value(['train', 'lr'], args.lr),
        weight_decay=get_config_value(['train', 'weight_decay'], args.weight_decay),
        grad_clip_norm=get_config_value(['train', 'grad_clip_norm'], args.grad_clip_norm),
        
        # Training
        seed=get_config_value(['train', 'seed'], args.seed),
        device=get_config_value(['train', 'device'], args.device),
        amp=get_config_value(['train', 'amp'], not args.no_amp),
        log_interval=get_config_value(['train', 'log_interval'], args.log_interval),
        out_dir=get_config_value(['train', 'out_dir'], args.out_dir),
        
        # Evaluation
        eval_every_steps=get_config_value(['policy', 'eval_every_steps'], args.eval_every_steps),
        eval_num_inference_steps=get_config_value(['policy', 'eval_num_inference_steps'], args.eval_num_inference_steps),
        max_eval_batches=get_config_value(['policy', 'max_eval_batches'], args.max_eval_batches),
        save_checkpoint_every=get_config_value(['train', 'save_checkpoint_every'], args.save_checkpoint_every),
        best_metric=get_config_value(['policy', 'metric'], args.best_metric),
        
        # Logging
        use_tensorboard=get_config_value(['logging', 'use_tensorboard'], not args.no_tensorboard),
        save_plots=get_config_value(['logging', 'save_plots'], not args.no_plots),
        verbose_logging=get_config_value(['logging', 'verbose'], args.verbose),
    )


def main():
    """Main entry point for training."""
    parser = build_argparser()
    args = parser.parse_args()
    
    # Merge configurations
    config = _merge_configurations(args)
    
    # Start training
    train(config)


if __name__ == '__main__':
    main()