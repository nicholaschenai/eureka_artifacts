@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Recalibrated maximum speed to achieve wider reward dispersion
    max_speed = 16.0
    forward_reward = (forward_velocity.clamp(min=0) / max_speed)

    # Enhance sideways penalty computation for meaningful alternation
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    optimized_sideways_penalty = -0.7 * (sideways_velocity / max_speed)

    # Amplify transformation parameters for better reward scaling
    temperature_forward = 2.0
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0
    
    temperature_sideway = 1.0
    transformed_sideways_penalty = torch.exp(temperature_sideway * optimized_sideways_penalty) - 1.0

    # Adjusted total reward for balance and desired task orientation
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "optimized_sideways_penalty": optimized_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
