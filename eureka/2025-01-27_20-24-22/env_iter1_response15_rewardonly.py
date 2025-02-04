@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   episode_length: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Adjusted Reward for distance to the handle of the door
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.5  # Increase temperature to make reward more noticeable
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature) * 10.0  # Scale up

    # Scaled down Reward for opening the door
    door_opening_goal_pos = 1.0  # hypothetical goal position (open position)
    door_opening_distance = torch.abs(cabinet_dof_pos[:, 3] - door_opening_goal_pos)
    door_opening_reward_temperature = 0.2  # Adjust temperature
    door_opening_reward = torch.exp(-door_opening_distance / door_opening_reward_temperature) * 5.0  # Scale down

    # Penalty for longer episodes to encourage quicker completion
    max_episode_length = 500.0
    time_penalty = (episode_length / max_episode_length) * -1.0  # Negative reward for taking too long

    # Total reward calculation with adjustments
    total_reward = distance_reward + door_opening_reward + time_penalty

    # Return the total reward and a breakdown of the individual components
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'time_penalty': time_penalty
    }
    return total_reward, reward_dict
