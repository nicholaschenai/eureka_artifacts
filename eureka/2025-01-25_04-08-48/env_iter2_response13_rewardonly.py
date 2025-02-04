@torch.jit.script
def compute_reward(root_states: torch.Tensor, up_vec: torch.Tensor, velocity: torch.Tensor, ang_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute linear velocity in the forward direction
    forward_velocity = velocity[:, 0]

    # Rescale forward_velocity to enhance sensitivity
    velocity_temp = 0.5
    transformed_velocity_reward = torch.exp(forward_velocity * velocity_temp) - 1.0

    # Compute the angular stability based on deviation of the vertical axis
    stability_proj = up_vec[:, 2]
    stability_reward = stability_proj
    stability_temp = 5.0
    transformed_stability_reward = torch.exp(-(1.0 - stability_reward) * stability_temp)

    # Aggregate rewards with weights adjusted for better balancing
    total_reward = 1.0 * transformed_velocity_reward + 0.2 * transformed_stability_reward

    # Reward dictionary for tracking the individual components
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_stability_reward": transformed_stability_reward,
    }

    return total_reward, reward_dict
