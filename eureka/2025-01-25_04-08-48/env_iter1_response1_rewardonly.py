@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, ang_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # The speed component of the reward function, scaled down
    velocity_reward = (potentials - prev_potentials) * 0.1

    # Stability reward redefined to include a penalty for high angular velocity
    uprightness = up_vec[:, 2]
    angular_penalty = torch.norm(ang_velocity, p=2, dim=-1)
    stability_reward = uprightness - 0.05 * angular_penalty

    # Check effective range of stability reward
    stability_reward = torch.clamp(stability_reward, min=0.0)

    # Improved transformation with adjusted temperature parameters
    velocity_temp = 0.5  # Decreased temperature for velocity reward to reduce dominance
    stability_temp = 5.0  # Increased temperature for stability reward to enhance impact

    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0
    transformed_stability_reward = torch.exp(stability_reward * stability_temp) - 1.0

    # Final reward as a weighted sum of transformed components
    total_reward = transformed_velocity_reward + transformed_stability_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "velocity_reward": transformed_velocity_reward,
        "stability_reward": transformed_stability_reward
    }

    return total_reward, reward_dict
