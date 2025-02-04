@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Adjust distance reward for improved balance
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 0.3  # Lowered temperature to reduce impact
    distance_reward = 1.0 - torch.tanh(temp_distance * distance_to_drawer)

    # Enhance open_reward by scaling to encourage more drawer movement
    open_pos_factor = torch.clip(cabinet_dof_pos[:, 3], min=0.0, max=1.0)
    temp_open = 1.0  # Increase temperature to elevate the reward impact
    open_reward = torch.tanh(temp_open * open_pos_factor)

    # Task success should become clear with open_reward sufficiently guiding to completion

    # Weight assignment adjusted for balance
    weight_distance = 0.2  # Reduced to lessen the follow impact
    weight_open = 0.8  # Increased, pressing importance on opening
    total_reward = weight_distance * distance_reward + weight_open * open_reward
    total_reward = torch.clamp(total_reward, min=0.0)

    # Components breakdown
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward
    }

    return total_reward, reward_components
