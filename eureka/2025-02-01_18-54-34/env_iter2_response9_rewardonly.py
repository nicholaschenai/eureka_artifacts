@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Scaling maximum speed to accommodate agent's potential
    max_speed = 12.0
    forward_reward = (forward_velocity / max_speed).clamp(0, 1)

    # Maintain or slightly scale sideways penalty
    sideways_velocity = torch.norm(velocity[:, 1:], p=2, dim=-1)
    sideways_penalty = -(sideways_velocity / max_speed).clamp(0, 1)

    # Exclude the heading reward due to non-contributory nature

    # Introduce new temperature variables for fine-tuning
    temperature1 = 1.5  # Adjusted to intensify focus on speed
    temperature2 = 0.5  # Potentially increased marginal penalty
  
    # Transform and scale rewards and penalties
    transformed_forward_reward = torch.exp(temperature1 * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature2 * sideways_penalty) - 1.0
    
    # Total reward definition with increased forward bias
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
