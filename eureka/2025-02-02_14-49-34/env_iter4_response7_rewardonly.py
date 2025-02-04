@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate distance reward with adjusted weight
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 0.3  # previously 0.5
    distance_reward = 1.0 - torch.tanh(temp_distance * distance_to_drawer)

    # Enhanced open reward based on drawer position
    open_pos_factor = torch.clip((cabinet_dof_pos[:, 3] - 0.5) * 2.0, min=0.0, max=1.0) 
    temp_open = 1.0  # increased to emphasize drawer movement
    open_reward = torch.tanh(temp_open * open_pos_factor)

    # Improved completion bonus for opening the drawer significantly
    temp_completion = 1.0
    completion_bonus = torch.where(cabinet_dof_pos[:, 3] > 0.8, torch.tensor(2.0, device=cabinet_dof_pos.device), torch.tensor(0.0, device=cabinet_dof_pos.device))
    completion_bonus = torch.tanh(temp_completion * completion_bonus)

    # Overall score with adjusted weights for balance
    weight_distance = 0.1  # reduced to give more importance to opening
    weight_open = 0.6   # increased for promoting open door motivation
    weight_completion = 0.3  # promotes completion success

    total_reward = weight_distance * distance_reward + weight_open * open_reward + weight_completion * completion_bonus
    total_reward = torch.clamp(total_reward, min=0.0)

    # Components breakdown
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "completion_bonus": completion_bonus
    }
    
    return total_reward, reward_components
