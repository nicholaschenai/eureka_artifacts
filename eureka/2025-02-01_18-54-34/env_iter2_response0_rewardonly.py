@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    max_speed = 15.0
    forward_reward = forward_velocity.clamp(min=0) / max_speed

    # Modify sideways penalty to improve significance for RL agent
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    optimized_sideways_penalty = -0.5 * sideways_velocity / max_speed

    # Adjust rewards to balance emphasis and amplify results
    temperature_forward = 1.5
    transformed_forward_reward = torch.exp(temperature_forward * forward_reward) - 1.0
    
    # Alter sideways penalty transformation for greater penalties
    temperature_sideway = 0.5
    transformed_sideways_penalty = torch.exp(temperature_sideway * optimized_sideways_penalty) - 1.0

    # Recalculate total reward, focusing on forward movement with balanced constraints
    total_reward = transformed_forward_reward + transformed_sideways_penalty

    return total_reward, {
        "forward_reward": forward_reward,
        "optimized_sideways_penalty": optimized_sideways_penalty,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
