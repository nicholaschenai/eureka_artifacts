@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    max_speed = 15.0  # Cap forward speed for reward calculations
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Increasing the side penalty to effectively counter non-forward motion
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    enhanced_sideways_penalty = -2.0 * sideways_velocity / max_speed  # Heavier penalty weight

    # Apply transformations with fine-tuned temperatures to improve reward quality
    temperature_forward = 1.7
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    temperature_sideway = 0.5
    # Rescale or remap the penalty impact for better differentiation, using negative exp
    transformed_sideways_penalty = -torch.exp(temperature_sideway * enhanced_sideways_penalty) + 1.0

    # Aggregate total reward for a well-balanced contribution of speed and trajectory control
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "enhanced_sideways_penalty": enhanced_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
