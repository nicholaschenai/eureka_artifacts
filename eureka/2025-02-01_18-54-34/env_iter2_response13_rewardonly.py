@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    max_speed = 10.0
    forward_reward = forward_velocity / max_speed

    # Scale sideways penalty to penalize effectively but not harshly
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -sideways_velocity / max_speed

    # Remove or redefine heading reward component since it is constant
    # We discard it for this implementation

    # Adjust temperature parameters for reward transformation
    temperature_forward = 0.5
    temperature_sideways = 0.5
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideways * sideways_penalty) - 1.0

    # Compute total reward with emphasis on speed and maintenance of stability
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
