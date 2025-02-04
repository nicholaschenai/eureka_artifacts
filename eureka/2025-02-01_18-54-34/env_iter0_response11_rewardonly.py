@torch.jit.script
def compute_reward(velocity: torch.Tensor, to_target: torch.Tensor, ang_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the forward velocity component
    forward_vel = torch.norm(velocity, p=2, dim=-1)

    # Main reward: Encourage high forward speed
    speed_reward = forward_vel

    # Penalty for deviation from target direction
    direction_penalty = torch.norm(to_target, p=2, dim=-1)

    # Penalty for angular velocity to encourage stable running
    stability_penalty = torch.norm(ang_velocity, p=2, dim=-1)

    # Combine the main reward with penalties
    speed_temperature = 0.1
    direction_temperature = 0.1
    stability_temperature = 0.1

    transformed_speed_reward = torch.exp(speed_reward / speed_temperature)
    transformed_direction_penalty = torch.exp(-direction_penalty / direction_temperature)
    transformed_stability_penalty = torch.exp(-stability_penalty / stability_temperature)

    total_reward = transformed_speed_reward * transformed_direction_penalty * transformed_stability_penalty

    # Create a reward components dictionary
    reward_components = {
        "speed_reward": transformed_speed_reward,
        "direction_penalty": transformed_direction_penalty,
        "stability_penalty": transformed_stability_penalty
    }

    return total_reward, reward_components
