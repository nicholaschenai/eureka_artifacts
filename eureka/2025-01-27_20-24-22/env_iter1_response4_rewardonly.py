@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Adjusted distance reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.5  # Increased temperature for more sensitivity
    distance_reward = 1.0 - torch.clamp(distance_to_handle / 1.0, 0.0, 1.0)  # Normalize distance and invert it
    
    # Adjusted door opening reward
    door_opening_reward_temperature = 1.0  # Scaled temperature for range adjustment
    normalized_door_pos = torch.clamp(cabinet_dof_pos[:, 3] / torch.tensor(0.5), 0.0, 1.0)  # Assuming max opening is 0.5
    door_opening_reward = torch.exp(normalized_door_pos / door_opening_reward_temperature)
    
    # Combine rewards, preferring normalized and balanced scaling
    total_reward = (distance_reward + 0.1 * door_opening_reward)  # Scaling down door opening component

    # Return the total reward and each component
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward * 0.1
    }
    return total_reward, reward_dict
