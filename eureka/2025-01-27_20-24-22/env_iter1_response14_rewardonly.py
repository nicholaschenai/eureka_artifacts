@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor,
                   drawer_grasp_pos: torch.Tensor,
                   cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Adjust distance reward sensitivity
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.05  # Reduced for higher sensitivity
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature) - 0.5  # Introduce a penalty component for being far
    
    # New scaling applied to door opening reward
    door_open_percent = cabinet_dof_pos[:, 3] / 1.0  # Normalize door position by a reasonable max opening value
    door_opening_reward_temperature = 0.2
    door_opening_reward = torch.exp(door_open_percent / door_opening_reward_temperature)

    # Encourage faster task completion with time penalty
    time_penalty = -0.01  # Add a small penalty per timestep to encourage quicker completion

    # Total reward calculation
    total_reward = distance_reward + door_opening_reward + time_penalty

    # Return the total reward and a breakdown of the individual components
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'time_penalty': time_penalty
    }
    return total_reward, reward_dict
