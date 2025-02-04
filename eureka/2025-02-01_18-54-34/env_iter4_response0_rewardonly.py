@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    
    # Calculate forward velocity
    forward_velocity = velocity[:, 0]
    max_speed = 15.0  # max normalized speed for scaling purposes
    
    # Forward Reward: boost scaling further for higher impact
    forward_reward = forward_velocity.clamp(min=0) / max_speed
    temperature_forward = 1.8
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    # Rewriting sideways penalty to inject higher variance
    angle_deviation = torch.norm(velocity[:, 1:3], p=2, dim=-1)  # Penalize yaw along with sideways motion
    rewritten_sideways_penalty = -angle_deviation / max_speed  # Normalize with max_speed
    temperature_sideways = 0.7
    transformed_sideways_penalty = torch.exp(temperature_sideways * rewritten_sideways_penalty) - 1.0
    
    # New Total Reward
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "rewritten_sideways_penalty": rewritten_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
