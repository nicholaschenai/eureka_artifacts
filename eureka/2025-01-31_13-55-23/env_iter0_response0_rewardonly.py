@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, velocities: torch.Tensor, heading_proj: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward velocity (assuming forward is along the positive X-axis)
    forward_velocity = velocities[:, 0]

    # Calculate distance to target in the lateral plane (X, Y)
    torso_position = root_states[:, 0:3]
    distance_to_target = torch.norm(targets[:, 0:2] - torso_position[:, 0:2], p=2, dim=-1)

    # Reward components for forward velocity and distance to target
    # Assuming the heading_proj is a cosine measure of heading alignment with target direction
    
    # Reward for moving forward quickly
    reward_forward_velocity = forward_velocity
    
    # Reward for maintaining alignment with target direction
    reward_heading_alignment = heading_proj

    # Penalize the distance to target to encourage reaching the target destination
    reward_distance_to_target = -distance_to_target

    # Temperatures for normalizing the reward components
    temp_forward_vel = 2.0  # Temperature for forward velocity component
    temp_heading_alignment = 1.0  # Temperature for heading alignment component
    temp_distance_to_target = 0.1  # Temperature for distance to target component

    # Apply transformation to each reward component
    transformed_reward_forward_velocity = torch.exp(temp_forward_vel * reward_forward_velocity)
    transformed_reward_heading_alignment = torch.exp(temp_heading_alignment * reward_heading_alignment)
    transformed_reward_distance_to_target = torch.exp(temp_distance_to_target * reward_distance_to_target)

    # Total reward: combination of the components
    total_reward = transformed_reward_forward_velocity + transformed_reward_heading_alignment + transformed_reward_distance_to_target

    # Reward components dictionary
    reward_components = {
        "reward_forward_velocity": transformed_reward_forward_velocity,
        "reward_heading_alignment": transformed_reward_heading_alignment,
        "reward_distance_to_target": transformed_reward_distance_to_target
    }

    return total_reward, reward_components
