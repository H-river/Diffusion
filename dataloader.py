import os
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NPZWaypointDataset(Dataset):
    """
    Modified dataset loader for your NPZ data format:
      - 'logits': 1x1616x1232 float32 feature map (ready to use)
      - 'waypoints': 20x2 float32 waypoints (x, y)  ### CHANGED: 20x3 -> 20x2
      - 'goal_local': 2 float32 goal position (x, y)
    
    Main modifications:
      1. Use logits directly as input map
      2. Apply sigmoid normalization to logits (values 0-1)
      3. Add data validation and error handling
      4. Add logging
    """

    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (1616, 1232),
        num_waypoints: int = 20,
        patch_size: int = 8,
        xy_scale: float = 1.0,
        # yaw_scale: float = np.pi, ### REMOVED: Argument no longer needed
        verbose: bool = False
    ):
        super().__init__()
        self.paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.paths.sort()
        
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")
        
        logger.info(f"Loaded {len(self.paths)} files from {data_dir}")
        
        self.H, self.W = image_size
        self.ps = patch_size
        self.K = num_waypoints
        self.xy_scale = xy_scale
        # self.yaw_scale = yaw_scale ### REMOVED
        self.verbose = verbose
        
        # Verify dimensions are divisible by patch_size
        if self.H % self.ps != 0 or self.W % self.ps != 0:
            logger.warning(
                f"Image size ({self.H}, {self.W}) is not divisible by patch_size {self.ps}. "
                f"Consider adjusting patch_size or using padding."
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            with np.load(p) as data:
                # Load your data format
                logits = data['logits'].astype(np.float32)  # Shape: (1, 1616, 1232)
                waypoints = data['waypoints'].astype(np.float32)  # Shape: (20, 3)
                goal = data['goal_local'].astype(np.float32)  # Shape: (2,)
                
            # Validate data shapes
            if logits.shape != (1, self.H, self.W):
                logger.warning(
                    f"Logits shape mismatch in {os.path.basename(p)}: "
                    f"expected (1, {self.H}, {self.W}), got {logits.shape}"
                )
                # Optional: Resize to expected shape
                logits = F.interpolate(
                    torch.from_numpy(logits).unsqueeze(0), 
                    size=(self.H, self.W), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0).numpy()
            
            if waypoints.shape[0] < self.K:
                raise ValueError(
                    f"Sample {os.path.basename(p)} has {waypoints.shape[0]} waypoints, "
                    f"expected >= {self.K}"
                )
            
            # Take only first K waypoints
            # CHANGED: Slice both K waypoints AND the first 2 dimensions (x, y)
            waypoints = waypoints[:self.K, :2]
            
            # Normalize waypoints and goal
            waypoints[:, :2] = waypoints[:, :2] / max(self.xy_scale, 1e-8)
            # waypoints[:, 2] = waypoints[:, 2] / max(self.yaw_scale, 1e-8) ### REMOVED: No 3rd dim to normalize
            
            goal = goal / max(self.xy_scale, 1e-8)
            
            # Convert logits to tensor and apply sigmoid normalization (0-1 range)
            map_tensor = torch.from_numpy(logits).float()  # Shape: [1, H, W]
            map_tensor = torch.sigmoid(map_tensor)  # Apply sigmoid normalization (0-1 range)
            
            # Optional: Add data statistics logging (only in verbose mode)
            if self.verbose and idx == 0:
                logger.info(f"Sample {os.path.basename(p)} statistics:")
                logger.info(f"  Map shape: {map_tensor.shape}")
                logger.info(f"  Map range after sigmoid: [{map_tensor.min():.3f}, {map_tensor.max():.3f}]")
                logger.info(f"  Waypoints shape: {waypoints.shape}")
                logger.info(f"  Goal: {goal}")
            
            return {
                'map': map_tensor,  # [1, H, W] with values in (0, 1) range
                'waypoints': torch.from_numpy(waypoints).float(),  # [K, 2]
                'goal': torch.from_numpy(goal).float(),  # [2]
            }
            
        except Exception as e:
            logger.error(f"Error loading {p}: {str(e)}")
            # Return empty data or skip (here we try the next sample)
            return self.__getitem__((idx + 1) % len(self.paths))


def create_logging_dir(base_dir: str) -> str:
    """Create logging directory and return path"""
    import time
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup file logging
    file_handler = logging.FileHandler(os.path.join(log_dir, "dataloader.log"))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return log_dir