@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Updated maximum speed for scaling, perhaps agent can go faster
    max_speed = 15.0
    forward_reward = forward_velocity / max_speed

    # Reevaluate and scale penalty components
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -sideways_velocity / max_speed

    # We can choose to drop the constant heading reward or rework it
    heading_proj = torch.ones_like(forward_velocity)

    # Combined reward with adjusted transformation scales
    temperature1 = 1.0
    temperature2 = 0.25
    transformed_forward_reward = torch.exp(temperature1 * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature2 * sideways_penalty) - 1.0

    # Update total reward with bias towards forward movement
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    # Return and track individual components
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "heading_reward": heading_proj,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
