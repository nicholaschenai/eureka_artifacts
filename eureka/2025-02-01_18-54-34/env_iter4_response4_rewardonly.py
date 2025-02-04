@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Normalize the forward velocity against a maximum speed
    max_speed = 15.0  
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Redefine sideways penalties for better gradient and impact
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    redesigned_sideways_penalty = -1.0 * sideways_velocity / max_speed
    
    # Introduce penalty for deviation from the direction to the target
    direction_vec = velocity / (torch.norm(velocity, p=2, dim=-1, keepdim=True) + 1e-6)
    forward_direction = torch.tensor([1.0, 0.0, 0.0], device=device)
    deviation_penalty = 1.0 - torch.dot(direction_vec, forward_direction.unsqueeze(0).expand_as(direction_vec))
    path_deviation_penalty = -0.5 * deviation_penalty
    
    # Utilize temperature parameters to fine tune encouraged/discouraged behaviors
    temperature_forward = 1.2
    temperature_sideway = 0.8
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideway * redesigned_sideways_penalty) - 1.0
    
    # Aggregate total reward
    total_reward = (
        transformed_forward_reward + 
        transformed_sideways_penalty + 
        path_deviation_penalty
    )

    return total_reward, {
        "forward_reward": forward_reward,
        "redesigned_sideways_penalty": redesigned_sideways_penalty,
        "path_deviation_penalty": path_deviation_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
