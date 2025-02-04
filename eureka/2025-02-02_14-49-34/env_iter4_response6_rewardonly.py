@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate distance to drawer as before
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 0.3
    distance_reward = 1.0 - torch.exp(-temp_distance * distance_to_drawer)

    # Improving the drawer opening reward with better scaling
    open_pos_factor = torch.clip(cabinet_dof_pos[:, 3], min=0.0, max=1.0)
    temp_open = 2.0  # Experimented higher value to accentuate progress
    open_reward = torch.sigmoid(temp_open * open_pos_factor)

    # Introduce task completion reward more clearly
    completion_threshold = 0.9
    completion_bonus = torch.where(open_pos_factor > completion_threshold, torch.tensor(1.0, device=open_pos_factor.device), torch.tensor(0.0, device=open_pos_factor.device))
    temp_completion = 5.0  # Provide clear strong incentive for completion
    completion_bonus = temp_completion * completion_bonus

    # Encouraging efficiency - reducing episode length
    efficiency_penalty = -torch.tanh(0.01 * torch.norm(cabinet_dof_vel, dim=-1))
    
    # Adjust weights accordingly
    weight_distance = 0.1
    weight_open = 0.3
    weight_completion = 0.5
    weight_efficiency = 0.1

    # Summing up total_reward
    total_reward = (weight_distance * distance_reward + weight_open * open_reward + 
                    weight_completion * completion_bonus + weight_efficiency * efficiency_penalty)
    total_reward = torch.clamp(total_reward, min=0.0)

    # Components breakdown for tracking
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "completion_bonus": completion_bonus,
        "efficiency_penalty": efficiency_penalty
    }
    
    return total_reward, reward_components
