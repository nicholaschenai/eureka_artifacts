@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Max speed for normalization
    max_speed = 15.0
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Adjust sideways penalty to include angle deviation for better control
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -0.5 * sideways_velocity / max_speed

    # Introducing angle deviation penalty
    angular_velocity = root_states[:, 10:13].norm(p=2, dim=-1)
    angle_deviation_penalty = -0.5 * angular_velocity.clamp(min=0, max=5.0) / 5.0

    # Enhance reward transformation for sideway and angle penalties
    temperature_forward = 1.5
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    temperature_sideway = 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideway * sideways_penalty) - 1.0

    temperature_angle = 0.5
    transformed_angle_deviation_penalty = torch.exp(temperature_angle * angle_deviation_penalty) - 1.0

    # Re-compute the total reward with refined components
    total_reward = (
        transformed_forward_reward + transformed_sideways_penalty + transformed_angle_deviation_penalty
    )

    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "angle_deviation_penalty": angle_deviation_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty,
        "transformed_angle_deviation_penalty": transformed_angle_deviation_penalty
    }
