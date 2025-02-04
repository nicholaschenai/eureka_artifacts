@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, torso_position: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Normalize the velocity reward by scaling
    velocity_change = potentials - prev_potentials
    velocity_reward = torch.clamp(velocity_change, max=1.0)  # Normalization by clamping

    # Revise stability reward: encourage keeping the torso above a specific height (e.g., 1.0)
    stability_temp = 0.5
    desired_height = 1.0
    height_diff = torch.abs(torso_position[:, 2] - desired_height)
    stability_reward = torch.exp(-height_diff * stability_temp)

    # Penalize falling down by checking if up vector is pointing significantly downward
    fall_penalty = torch.where(up_vec[:, 2] < 0.5, -1.0, 0.0)  # Penalize if torso tilts too much

    # Combine the rewards and penalties
    total_reward = velocity_reward + stability_reward + fall_penalty

    # Collect reward components
    reward_dict = {
        "velocity_reward": velocity_reward,
        "stability_reward": stability_reward,
        "fall_penalty": fall_penalty
    }

    return total_reward, reward_dict
