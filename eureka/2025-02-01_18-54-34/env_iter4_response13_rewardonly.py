@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    
    # Extract forward velocity and sideways velocity components
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    sideways_velocity = velocity[:, 1:3]
    
    # Define reasonable limits for scaling velocity components
    max_speed = 15.0

    # Reward component: Encourage maximum forward velocity
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Sideways penalty rewritten: Discourage any sideways motion
    side_deviation_penalty = -torch.norm(sideways_velocity, p=2, dim=-1) / max_speed

    # Temperature for forward velocity reward scaling
    temperature_forward = 1.5
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0

    # Adjust temperature on sideways penalty to apply greater penalty for side movements
    temperature_sideway = 0.8
    transformed_side_deviation_penalty = torch.exp(temperature_sideway * side_deviation_penalty) - 1.0

    # Combine reward components with more weight on forward rewards
    total_reward = 2.0 * transformed_forward_reward + transformed_side_deviation_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "side_deviation_penalty": side_deviation_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_side_deviation_penalty": transformed_side_deviation_penalty
    }
