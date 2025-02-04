@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward (x-direction) velocity; assuming x-axis represents forward movement
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Increased weight for moving forward
    max_speed = 10.0
    forward_reward = 1.5 * (forward_velocity / max_speed)
    
    # More penalization for sideways and backward movement
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -2.0 * (sideways_velocity / max_speed)
    
    # Transform forwards and sideways components for better gradients
    temperature1 = 0.5
    temperature2 = 0.5
    transformed_forward_reward = torch.exp(temperature1 * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature2 * sideways_penalty) - 1.0

    # Total reward with a higher focus on forward movement and sideways penalty
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    # Return the total reward and a dictionary of reward components
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
