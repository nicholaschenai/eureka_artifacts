@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Adjusted Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.2  # Decreased further for more gradient
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Rescaled Door Opening Reward
    door_opening_reward_temperature = 0.5  # Increased to tone down the magnitude
    door_opening_reward = 1.0 / (1.0 + torch.exp(-cabinet_dof_pos[:, 3] / door_opening_reward_temperature))

    # Introduced Energy Penalty as Movement Penalty (based on velocity)
    energy_penalty_temperature = 0.1
    energy_penalty = torch.exp(-torch.square(cabinet_dof_vel[:, 3]) / energy_penalty_temperature)

    # Total reward calculation with better weighting
    total_reward = 0.4 * distance_reward + 0.4 * door_opening_reward - 0.2 * energy_penalty

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'energy_penalty': energy_penalty
    }
    return total_reward, reward_dict
