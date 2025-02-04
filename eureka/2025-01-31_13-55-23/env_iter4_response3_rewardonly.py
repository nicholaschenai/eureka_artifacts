@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity and compute forward velocity (positive x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Emphasized forward velocity reward
    forward_velocity_scale = 2.5  # Increased scale for stronger emphasis
    forward_velocity_reward = forward_velocity_scale * forward_velocity

    # Adjusted energy penalty scale and temperature
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 1.5  # Increased temperature for stronger impact
    energy_penalty_scaled = -energy_temp * energy_penalty

    # Remove the stable motion penalty as it might conflict with achieving higher speeds

    # Total reward with adjusted temperature parameter
    overall_temp = 0.15
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled))

    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
