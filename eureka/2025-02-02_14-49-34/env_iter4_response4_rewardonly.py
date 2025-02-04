@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Adjust distance reward
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 0.5
    distance_reward = 1.0 - torch.tanh(temp_distance * distance_to_drawer)

    # Enhance and transform open reward more aggressively
    open_pos_factor = torch.clip(cabinet_dof_pos[:, 3], min=0.0, max=1.0)
    temp_open = 1.0
    open_reward = torch.tanh(temp_open * open_pos_factor)
    
    # Replace or adjust completion bonus reward to include intermediary progress
    intermediate_progress = (open_pos_factor - 0.5) * 2.0
    temp_progress = 0.4
    progressive_reward = torch.tanh(temp_progress * intermediate_progress)
    
    # Bonus for full completion
    temp_completion = 2.0
    completion_bonus = torch.where(cabinet_dof_pos[:, 3] > 0.9, torch.tensor(1.0, device=cabinet_dof_pos.device), torch.tensor(0.0, device=cabinet_dof_pos.device))
    completion_bonus = torch.tanh(temp_completion * completion_bonus)

    # Total combining new balance and weight emphasis
    weight_distance = 0.2
    weight_open = 0.3
    weight_progress = 0.3
    weight_completion = 0.2
    total_reward = weight_distance * distance_reward + weight_open * open_reward + weight_progress * progressive_reward + weight_completion * completion_bonus
    total_reward = torch.clamp(total_reward, min=0.0)

    # Components breakdown
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "progressive_reward": progressive_reward,
        "completion_bonus": completion_bonus
    }
    
    return total_reward, reward_components
