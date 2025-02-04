@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Re-calibrate distance reward with a new temperature
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 0.3  # Decrease temperature for more sensitivity at close distances
    distance_reward = 1.0 - torch.exp(-temp_distance * distance_to_drawer)

    # Re-focus open drawer reward
    open_pos_factor = torch.clip(cabinet_dof_pos[:, 3], min=0.0, max=1.0)
    temp_open = 0.8  # Increased temperature to emphasize changes
    open_reward = torch.exp(temp_open * (open_pos_factor - 0.5))  # Ensure range transformation

    # Simplified task completion recognition for training phase only
    completion_reward = torch.tensor(0.0, device=cabinet_dof_pos.device)
    
    # Episode penalty to encourage efficient task completion
    penalty_factor = torch.tensor(0.1, device=cabinet_dof_pos.device)
    efficiency_penalty = penalty_factor * distance_to_drawer

    # Total reward setup
    weight_distance = 0.2
    weight_open = 0.7
    weight_penalty = 0.1
    total_reward = weight_distance * distance_reward + weight_open * open_reward - weight_penalty * efficiency_penalty
    total_reward = torch.clamp(total_reward, min=0.0)  # Ensure non-negative total reward

    # Components breakdown
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "efficiency_penalty": efficiency_penalty
    }

    return total_reward, reward_components
