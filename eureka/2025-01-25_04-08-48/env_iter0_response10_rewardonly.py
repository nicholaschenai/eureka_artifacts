@torch.jit.script
def compute_reward(velocity: torch.Tensor, heading_proj: torch.Tensor, up_proj: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Encouraging high velocity in a positive forward direction
    speed_reward = torch.norm(velocity, p=2, dim=-1)
    
    # Adding a stability reward based on the alignment of the torso's up vector with the world up vector
    stability_reward = up_proj

    # Adding a heading reward based on the alignment with the target direction
    heading_reward = heading_proj

    # Applying transformations for better shaping of the above components
    temperature_speed = 1.0
    temperature_stability = 0.1
    temperature_heading = 0.1

    speed_reward_transformed = torch.exp(temperature_speed * (speed_reward - 1.0))
    stability_reward_transformed = torch.exp(temperature_stability * (stability_reward - 1.0))
    heading_reward_transformed = torch.exp(temperature_heading * (heading_reward - 1.0))

    # Total reward is a weighted sum of the component rewards
    total_reward = speed_reward_transformed + 0.1 * stability_reward_transformed + 0.1 * heading_reward_transformed

    # Reward components for analysis
    reward_components = {
        "speed_reward": speed_reward,
        "stability_reward": stability_reward,
        "heading_reward": heading_reward,
        "speed_reward_transformed": speed_reward_transformed,
        "stability_reward_transformed": stability_reward_transformed,
        "heading_reward_transformed": heading_reward_transformed
    }

    return total_reward, reward_components
