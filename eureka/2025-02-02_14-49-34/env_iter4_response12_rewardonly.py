@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Tuning distance reward: Lower weight, same transformation
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 0.3  # Reduced temperature
    distance_reward = 1.0 - torch.tanh(temp_distance * distance_to_drawer)

    # Enhance open reward: Adjust scaling and transformation
    open_pos_factor = torch.clip(cabinet_dof_pos[:, 3], min=0.0, max=1.0)
    temp_open = 1.0  # Increased temperature
    open_reward = torch.sigmoid(temp_open * open_pos_factor)

    # Remove or redefine completion_bonus for inefficiency
    # Completion bonus converted to a sharper exponential function for achievement encouragement
    temp_completion_bonus = 3.0  # High sharpness for completion reward
    completion_bonus = torch.exp(-temp_completion_bonus * (1 - open_pos_factor)) * torch.where(open_pos_factor > 0.9, torch.tensor(1.0, device=cabinet_dof_pos.device), torch.tensor(0.0, device=cabinet_dof_pos.device))
    
    # Sum calculated components - repurpose weights
    weight_distance = 0.2
    weight_open = 0.5
    weight_completion_bonus = 0.3
    total_reward = weight_distance * distance_reward + weight_open * open_reward + weight_completion_bonus * completion_bonus
    total_reward = torch.clamp(total_reward, min=0.0)

    # Components breakdown
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "completion_bonus": completion_bonus
    }
    
    return total_reward, reward_components
