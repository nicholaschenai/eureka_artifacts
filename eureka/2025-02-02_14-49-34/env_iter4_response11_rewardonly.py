@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Distance reward
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 1.0
    distance_reward = 1.0 - torch.tanh(temp_distance * distance_to_drawer)

    # Open drawer reward with adjusted scale
    open_pos_factor = torch.clip(cabinet_dof_pos[:, 3], min=0.0, max=1.0)
    temp_open = 2.0  # Increased temperature to enhance scaling and reward impact
    open_reward = torch.tanh(temp_open * open_pos_factor)

    # Revised completion bonus with adjusted criteria
    temp_completion = 5.0  # Increased temperature for strong reward signaling
    completion_bonus = torch.where(cabinet_dof_pos[:, 3] > 0.8, torch.exp(temp_completion * (cabinet_dof_pos[:, 3] - 0.8)) - 1.0, torch.tensor(0.0, device=cabinet_dof_pos.device))

    # Sum total reward with revised weights
    weight_distance = 0.1
    weight_open = 0.7
    weight_completion = 0.2  # Completion bonus weight to encourage completion
    total_reward = weight_distance * distance_reward + weight_open * open_reward + weight_completion * completion_bonus
    total_reward = torch.clamp(total_reward, min=0.0)

    # Components breakdown with adjusted scaling
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "completion_bonus": completion_bonus
    }
    
    return total_reward, reward_components
