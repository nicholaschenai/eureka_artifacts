@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   episode_length: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward for distance to the handle of the door
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.5  # Adjusted temperature to reduce saturation
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Reward for opening the door, rescaled
    door_opening_reward_temperature = 5.0  # Adjusted temperature for re-scaling
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # Penalty for episode length to encourage quicker task completion
    length_penalty_strength = 0.002  # Penalty per time step
    length_penalty = length_penalty_strength * episode_length

    # Total reward calculation
    total_reward = distance_reward + 0.1 * door_opening_reward - length_penalty

    # Return the total reward and a breakdown of the individual components
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'length_penalty': -length_penalty
    }
    return total_reward, reward_dict
