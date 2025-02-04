@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Velocity Reward
    velocity_reward = potentials - prev_potentials

    # Stability Reward - Rewriting to penalize larger deviations from upright posture
    stability_penalty = (1.0 - up_vec[:, 2]) ** 2
    stability_temp = 0.5  # Adjusting the impact of the stability penalty
    transformed_stability_penalty = -torch.exp(stability_penalty * stability_temp)

    # Smooth Movement Reward - Penalize high joint velocities to encourage smoother movements
    smoothness_temp = 0.1
    smoothness_penalty = torch.norm(dof_vel, p=2, dim=-1)
    transformed_smoothness_penalty = -torch.exp(smoothness_penalty * smoothness_temp)

    # Combine all components
    total_reward = velocity_reward + transformed_stability_penalty + transformed_smoothness_penalty

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "velocity_reward": velocity_reward,
        "stability_penalty": transformed_stability_penalty,
        "smoothness_penalty": transformed_smoothness_penalty
    }

    return total_reward, reward_dict
