@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Re-evaluate Distance to Handle Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    rescaled_distance_reward_temperature = 0.05
    rescaled_distance_reward = torch.exp(-distance_to_handle / rescaled_distance_reward_temperature)

    # Normalize Door Opening Reward
    door_opening_norm_temperature = 5.0
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / door_opening_norm_temperature)

    # Balance reward component magnitudes by normalizing each component
    scaled_distance_reward = rescaled_distance_reward * 2.0
    normalized_door_opening_reward = door_opening_reward / 10.0

    # Total reward calculation
    total_reward = scaled_distance_reward + normalized_door_opening_reward

    # Return the total reward and a breakdown of the individual components
    reward_dict = {
        'scaled_distance_reward': scaled_distance_reward,
        'normalized_door_opening_reward': normalized_door_opening_reward
    }
    return total_reward, reward_dict
