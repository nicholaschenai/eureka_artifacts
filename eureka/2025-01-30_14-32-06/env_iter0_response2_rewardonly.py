@torch.jit.script
def compute_reward(velocity: torch.Tensor, actions: torch.Tensor, up_proj: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward for forward velocity: encourage higher forward speed
    forward_velocity_reward = velocity[:, 0]  # Assuming x-axis is forward
    velocity_temp = 0.1
    forward_velocity_reward = torch.exp(forward_velocity_reward / velocity_temp)

    # Encourage stability using up projection
    stability_reward = up_proj
    stability_temp = 0.1
    stability_reward = torch.exp(stability_reward / stability_temp)

    # Penalize for excessive action (energy efficiency)
    action_penalty = torch.sum(actions**2, dim=-1)
    action_penalty_temp = 0.1
    action_penalty = torch.exp(-action_penalty / action_penalty_temp)

    # Total reward combines all components
    total_reward = forward_velocity_reward + stability_reward + action_penalty

    # Reward components to be returned for analysis
    reward_components = {
        "forward_velocity_reward": forward_velocity_reward,
        "stability_reward": stability_reward,
        "action_penalty": action_penalty,
    }

    return total_reward, reward_components
