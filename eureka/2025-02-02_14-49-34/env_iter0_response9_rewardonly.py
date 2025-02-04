@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = franka_grasp_pos.device
    
    # Reward component parameters
    distance_threshold = 0.02
    distance_reward_multiplier = 10.0
    open_reward_threshold = 0.5
    velocity_penalty_multiplier = -0.1
    
    # Calculate different reward components
    # 1. Distance reward to move hand closer to the drawer handle
    to_target_distance = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    distance_reward = torch.where(to_target_distance < distance_threshold, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    distance_reward *= distance_reward_multiplier

    # 2. Opening reward based on the cabinet DOF position 
    open_threshold = cabinet_dof_pos[:, 3] > open_reward_threshold
    open_reward = torch.where(open_threshold, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    
    # 3. Penalize high velocity to ensure stability when opening the drawer
    velocity_penalty = torch.norm(cabinet_dof_vel[:, 3].unsqueeze(-1), dim=-1) * velocity_penalty_multiplier

    # Total reward is the sum of individual components
    total_reward = distance_reward + open_reward + velocity_penalty
  
    # Create a reward dictionary to contain individual components
    reward_dict = {
        'distance_reward': distance_reward,
        'open_reward': open_reward,
        'velocity_penalty': velocity_penalty
    }
    
    return total_reward, reward_dict
