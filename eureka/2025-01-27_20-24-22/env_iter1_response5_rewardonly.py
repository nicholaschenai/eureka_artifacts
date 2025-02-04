@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor,
                   episode_length: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Modified distance reward to increase influence
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.5
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Scale down the door opening reward for better balance
    door_opening_scaling_factor = 0.1
    door_opening_reward = (cabinet_dof_pos[:, 3]) * door_opening_scaling_factor

    # Additional penalty to discourage lengthy episodes
    time_penalty_factor = -0.01
    time_penalty = episode_length * time_penalty_factor

    # Total reward calculation
    total_reward = distance_reward + door_opening_reward + time_penalty

    # Return the total reward and a breakdown of the individual components
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'time_penalty': time_penalty
    }
    return total_reward, reward_dict
