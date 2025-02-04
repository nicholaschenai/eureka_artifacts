@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # A more adjusted maximum speed for normalization
    max_speed = 20.0  # Increased given running task focus
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Revised sideways penalty with a stronger effect
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    redesigned_sideways_penalty = -1.0 * sideways_velocity / max_speed

    # Temperature adjustments for better regime
    temperature_forward = 1.5
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    # Augment sideways penalty transformation for clearer deterring effects
    temperature_sideway = 1.2
    transformed_sideways_penalty = torch.exp(temperature_sideway * redesigned_sideways_penalty) - 1.0

    # Aggregate rewards
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "redesigned_sideways_penalty": redesigned_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
