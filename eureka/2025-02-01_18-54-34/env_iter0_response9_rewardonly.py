@torch.jit.script
def compute_reward(velocity: torch.Tensor, actions: torch.Tensor, up_proj: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for transforming rewards
    velocity_temp = 0.1
    energy_temp = 0.1
    stability_temp = 0.1

    # Velocity Reward: encourage forward velocity
    forward_velocity_reward = velocity[:, 0]  # assuming the x-direction is forward
    forward_velocity_reward = torch.exp(velocity_temp * forward_velocity_reward) - 1

    # Energy Penalty: discourage high energy consumption
    energy_penalty = torch.norm(actions, p=2, dim=-1)
    energy_penalty = torch.exp(-energy_temp * energy_penalty)

    # Stability Bonus: maintain upright position
    stability_bonus = up_proj
    stability_bonus = torch.exp(stability_temp * stability_bonus)

    # Total reward
    total_reward = forward_velocity_reward + energy_penalty + stability_bonus

    # Return total reward and individual components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty": energy_penalty,
        "stability_bonus": stability_bonus
    }
    
    return total_reward, reward_dict
