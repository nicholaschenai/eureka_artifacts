@torch.jit.script
def compute_reward(
    torso_position: torch.Tensor, 
    velocity: torch.Tensor, 
    targets: torch.Tensor, 
    angle_to_target: torch.Tensor, 
    dt: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Compute direction to target
    to_target = targets - torso_position
    to_target[:, 2] = 0
    to_target_norm = torch.norm(to_target, dim=1, keepdim=True)
    
    # Compute velocity in the direction towards the target
    velocity_towards_target = torch.sum(velocity * (to_target / to_target_norm), dim=1)

    # Reward for moving quickly towards the target
    speed_reward = velocity_towards_target.clone()

    # Angular penalty - encourage the humanoid to face the target direction
    angle_penalty = -torch.abs(angle_to_target).view(-1)

    # Temperature parameters for transforming reward components
    speed_temp = 1.0
    angle_temp = 0.1
    
    # Transform reward components
    transformed_speed_reward = torch.exp(speed_reward / speed_temp)
    transformed_angle_penalty = torch.exp(angle_penalty / angle_temp)
    
    # Total reward is the sum of transformed components and potentials (for directional guidance)
    total_reward = transformed_speed_reward + transformed_angle_penalty
    
    # Gathering the components for debugging purposes
    reward_components = {
        'transformed_speed_reward': transformed_speed_reward,
        'transformed_angle_penalty': transformed_angle_penalty
    }
    
    return total_reward, reward_components
