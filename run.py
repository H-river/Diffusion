import argparse
import os
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from model import build_model
from train import NPZWaypointDataset, evaluate_sampling_performance


def load_checkpoint(ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    """Load model checkpoint."""
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Extract model state and config
    model_state = ckpt['model_state_dict']
    config = ckpt['config']
    metrics = ckpt.get('metrics', {})
    
    return {
        'model_state': model_state,
        'config': config,
        'metrics': metrics,
        'global_step': ckpt.get('global_step', 0),
        'epoch': ckpt.get('epoch', 0),
        'best_score': ckpt.get('best_score', float('inf')),
        'best_metric': ckpt.get('best_metric', 'ade')
    }


def build_model_from_config(config: Dict[str, Any], device: torch.device):
    """Build model from configuration dictionary."""
    print(f"Building model with configuration:")
    print(f"  Image size: {config['image_height']}x{config['image_width']}")
    print(f"  Patch size: {config['patch_size']}")
    print(f"  Num waypoints: {config['num_waypoints']}")
    print(f"  Embed dim: {config['embed_dim']}")
    print(f"  Depth: {config['depth']}")
    
    model = build_model(
        image_size=(config['image_height'], config['image_width']),
        patch_size=config['patch_size'],
        in_chans=1,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        num_waypoints=config['num_waypoints'],
        dropout=config['dropout'],
        attn_dropout=config.get('attn_dropout', 0.1),
        use_learned_img_pos=config.get('use_learned_img_pos', False),
        mode=config.get('mode', 'encoder_decoder'),
        causal_attn=config.get('causal_attn', False),
        time_as_cond=config.get('time_as_cond', True),
        obs_as_cond=config.get('obs_as_cond', False),
        cond_dim=config.get('cond_dim', 0),
        n_cond_layers=config.get('n_cond_layers', 0),
        vision_n_layers=config.get('vision_n_layers', 0),
    ).to(device)
    
    return model


def run_inference(
    ckpt_path: str,
    data_dir: str,
    batch_size: int = 8,
    num_inference_steps: int = 50,
    device: Optional[str] = None,
    max_batches: Optional[int] = None,
    save_preds: Optional[str] = None,
    verbose: bool = True,
):
    """Run inference on a dataset using a trained model."""
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"\n{'='*60}")
    print("INFERENCE RUN")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Inference steps: {num_inference_steps}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(ckpt_path, device)
    config = checkpoint['config']
    model_state = checkpoint['model_state']
    
    # Print checkpoint info
    print(f"Model epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Global step: {checkpoint.get('global_step', 'N/A')}")
    print(f"Best metric: {checkpoint.get('best_metric', 'N/A')}")
    print(f"Best score: {checkpoint.get('best_score', 'N/A'):.6f}")
    
    # Build model
    model = build_model_from_config(config, device)
    model.load_state_dict(model_state)
    model.eval()
    print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create diffusion scheduler for inference
    scheduler = DDPMScheduler(
        num_train_timesteps=config.get('num_diffusion_steps', 1000),
        beta_schedule=config.get('beta_schedule', 'squaredcos_cap_v2'),
        clip_sample=True,
        prediction_type='epsilon',
    )
    
    # Create dataset and dataloader
    print(f"\nLoading dataset from: {data_dir}")
    dataset = NPZWaypointDataset(
        data_dir=data_dir,
        image_size=(config['image_height'], config['image_width']),
        num_waypoints=config['num_waypoints'],
        patch_size=config['patch_size'],
        xy_scale=config['xy_scale'],
        # yaw_scale=config['yaw_scale'], ### CHANGED: REMOVED arg
        verbose=verbose,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Number of batches: {len(dataloader)}")
    
    # Run evaluation
    print(f"\n{'='*60}")
    print("RUNNING EVALUATION")
    print(f"{'='*60}")
    
    metrics = evaluate_sampling_performance(
        model=model,
        data_loader=dataloader,
        scheduler=scheduler,
        device=device,
        xy_scale=config['xy_scale'],
        # yaw_scale=config['yaw_scale'], ### CHANGED: REMOVED arg
        num_inference_steps=num_inference_steps,
        max_batches=max_batches if max_batches is not None else len(dataloader),
        verbose=verbose,
    )
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Average Displacement Error (ADE): {metrics['ade']:.4f} m")
    print(f"Final Displacement Error (FDE): {metrics['fde']:.4f} m")
    # print(f"Yaw Mean Absolute Error: {metrics['yaw_mae']:.4f} rad") ### CHANGED: REMOVED
    print(f"{'='*60}")
    
    # Save predictions if requested
    if save_preds:
        os.makedirs(save_preds, exist_ok=True)
        print(f"\nSaving predictions to: {save_preds}")
        
        # Configure scheduler for inference
        scheduler.set_timesteps(num_inference_steps)
        
        predictions_dir = os.path.join(save_preds, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Statistics
        total_samples = 0
        saved_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                # Load batch data with correct key names
                img = batch['map'].to(device)            # [B, 1, H, W]
                gt_waypoints = batch['waypoints']        # [B, K, 2] (Changed from 3)
                goal = batch['goal'].to(device)          # [B, 2]
                
                B = img.shape[0]
                total_samples += B
                
                # Initialize noise
                sample = torch.randn_like(gt_waypoints.to(device))
                
                # Pre-encode vision and goal tokens (once for efficiency)
                vision_tokens = model._forward_vision(img)
                goal_tokens = model._forward_goal(goal)
                
                # Denoising loop
                if verbose:
                    print(f"  Batch {batch_idx + 1}/{len(dataloader)}: Generating {B} predictions...")
                
                for i, t in enumerate(scheduler.timesteps):
                    # Predict noise
                    noise_pred = model._denoise(sample, t, vision_tokens, goal_tokens)
                    
                    # Update sample with scheduler
                    sample = scheduler.step(noise_pred, t, sample).prev_sample
                    
                    if verbose and i % 10 == 0 and batch_idx == 0:
                        print(f"    Denoising step {i}/{len(scheduler.timesteps)}")
                
                # Denormalize predictions
                pred_waypoints = sample.cpu().numpy()  # [B, K, 2]
                pred_waypoints[..., :2] *= config['xy_scale']
                # pred_waypoints[..., 2] *= config['yaw_scale'] ### CHANGED: REMOVED
                
                # Denormalize ground truth
                gt_waypoints_denorm = gt_waypoints.numpy().copy()
                gt_waypoints_denorm[..., :2] *= config['xy_scale']
                # gt_waypoints_denorm[..., 2] *= config['yaw_scale'] ### CHANGED: REMOVED
                
                # Save each sample
                for b in range(B):
                    # Get original file index
                    sample_idx = batch_idx * batch_size + b
                    
                    # Create prediction data
                    prediction_data = {
                        'predicted_waypoints': pred_waypoints[b],      # [K, 2]
                        'ground_truth_waypoints': gt_waypoints_denorm[b],  # [K, 2]
                        'goal': goal[b].cpu().numpy() * config['xy_scale'],  # [2]
                        'image_size': (config['image_height'], config['image_width']),
                        'config': config,
                    }
                    
                    # Save as NPZ
                    output_path = os.path.join(predictions_dir, f"prediction_{sample_idx:06d}.npz")
                    np.savez_compressed(output_path, **prediction_data)
                    saved_samples += 1
                
                if verbose and batch_idx % 10 == 0:
                    print(f"  Saved {saved_samples} predictions so far...")
        
        print(f"\nSaved {saved_samples}/{total_samples} predictions to {predictions_dir}")
        
        # Create a summary file
        summary = {
            'total_samples': total_samples,
            'saved_predictions': saved_samples,
            'metrics': metrics,
            'config': config,
            'checkpoint_info': {
                'path': ckpt_path,
                'epoch': checkpoint.get('epoch', 'N/A'),
                'global_step': checkpoint.get('global_step', 'N/A'),
                'best_metric': checkpoint.get('best_metric', 'N/A'),
                'best_score': checkpoint.get('best_score', 'N/A'),
            }
        }
        
        summary_path = os.path.join(save_preds, "inference_summary.npz")
        np.savez_compressed(summary_path, **summary)
        print(f"Summary saved to: {summary_path}")
    
    print(f"\nInference completed successfully!")
    print(f"{'='*60}")
    
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Run diffusion waypoint predictor inference on NPZ dataset'
    )
    
    # Required arguments
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to trained checkpoint (.pt file)')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to directory of .npz files for inference')
    
    # Inference parameters
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for inference (default: 8)')
    parser.add_argument('--num-inference-steps', type=int, default=50,
                       help='Number of denoising steps (default: 50)')
    
    # Device and processing
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run on (cuda/cpu, default: auto-detect)')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Maximum number of batches to process (default: all)')
    
    # Output options
    parser.add_argument('--save-preds', type=str, default=None,
                       help='Directory to save predicted waypoints (optional)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce verbose output')
    
    return parser


def main():
    """Main entry point for inference."""
    parser = build_argparser()
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.ckpt):
        print(f"ERROR: Checkpoint file not found: {args.ckpt}")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        return
    
    try:
        metrics = run_inference(
            ckpt_path=args.ckpt,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            device=args.device,
            max_batches=args.max_batches,
            save_preds=args.save_preds,
            verbose=not args.quiet,
        )
        
        # Exit with code 0 for success
        return 0
        
    except Exception as e:
        print(f"\nERROR during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)