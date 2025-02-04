@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                   
    # Enhanced reward for the proximity to handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 1.0
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Scaled door opening reward
    door_opening_reward_temperature = 0.5
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # Exploration encouragement to prevent stuck behavior.
    exploration_reward_temperature = 0.1
    exploration_reward = torch.exp(torch.norm(franka_grasp_pos, p=2, dim=-1) / exploration_reward_temperature)

    # Normalize individual rewards to bring them to comparable scales
    max_distance = torch.tensor(1.0, device=franka_grasp_pos.device)
    max_opening_angle = torch.tensor(1.0, device=cabinet_dof_pos.device)

    normalized_distance_reward = distance_reward / max_distance
    normalized_door_opening_reward = door_opening_reward / max_opening_angle
    normalized_exploration_reward = exploration_reward / max_distance

    # Total reward calculation
    total_reward = 0.5 * normalized_distance_reward + 1.0 * normalized_door_opening_reward + 0.1 * normalized_exploration_reward

    # Return the total reward and a breakdown of the individual components
    reward_dict = {
        'distance_reward': normalized_distance_reward,
        'door_opening_reward': normalized_door_opening_reward,
        'exploration_reward': normalized_exploration_reward
    }
    
    return total_reward, reward_dict
