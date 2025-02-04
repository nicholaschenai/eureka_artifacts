@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward velocity; assuming x-axis represents forward movement
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Reward for moving forward; increased emphasis on forward speed
    max_speed = 10.0
    forward_reward = forward_velocity / max_speed

    # Penalize sideways and backward movement
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -sideways_velocity / max_speed

    # Remove heading_reward since it's constant in this scenario

    # Combine rewards with new transformations
    temperature_forward = 0.3
    temperature_sideways = 0.7
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideways * sideways_penalty) - 1.0

    # Total reward, giving more weight to the forward movement
    weight_forward = 1.5
    weight_sideways = 0.5
    total_reward = weight_forward * transformed_forward_reward + weight_sideways * transformed_sideways_penalty

    # Return the total reward and a dictionary of reward components
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
