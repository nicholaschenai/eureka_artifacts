@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Forward Reward
    forward_reward = torch.clamp(forward_velocity, min=0)
    max_speed = 15.0  # reasonable forward speed and for normalization
    forward_reward /= max_speed

    # Redesigned Sideways Penalty: More significant for deviation control
    sideways_velocity = torch.norm(velocity[:, 1:], p=2, dim=-1)
    sideways_penalty = -sideways_velocity / max_speed

    # Apply transformations
    temperature_forward = 2.0
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    temperature_sideway = 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideway * sideways_penalty) - 1.0

    # Balance the components for the total reward
    total_reward = transformed_forward_reward + 1.5 * transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty,
    }
