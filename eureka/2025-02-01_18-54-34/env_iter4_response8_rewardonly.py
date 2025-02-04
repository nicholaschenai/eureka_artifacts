@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Improved forward reward scaling
    max_speed = 15.0
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Rethink sideways penalty: Penalize only substantial sideways motion
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    significant_sideways_threshold = 0.05
    enhanced_sideways_penalty = torch.where(
        sideways_velocity > significant_sideways_threshold,
        -sideways_velocity / max_speed,
        torch.zeros_like(sideways_velocity)
    )

    # Exponential scaling to reshape the reward gradients
    temperature_forward = 2.0
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    # Re-scaled axiswise penalty if necessary
    temperature_sideway = 1.2
    transformed_sideways_penalty = torch.exp(temperature_sideway * enhanced_sideways_penalty) - 1.0

    # Total reward recalibrated to balance strategic components
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "enhanced_sideways_penalty": enhanced_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
