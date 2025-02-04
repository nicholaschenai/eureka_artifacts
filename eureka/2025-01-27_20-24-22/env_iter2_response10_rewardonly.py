@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Enhanced Distance Reward with higher temperature sensitivity
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.1  # More sensitive to distance
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Adjusted Door Opening Reward for reduced magnitude
    door_opening_temp = 1.0  # Adjusted scaling
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3]) / door_opening_temp

    # Reformulated Movement Penalty with increased influence
    movement_threshold = 0.05
    movement_penalty_temperature = 1.0  # Reinforced influence
    movement_penalty = torch.exp(-torch.clamp(cabinet_dof_vel[:, 3], min=-movement_threshold, max=movement_threshold) / movement_penalty_temperature)

    # Penalty for longer episode lengths as a new component
    max_length = 500.0
    length_penalty = torch.exp(-cabinet_dof_pos[:, 3] / max_length)

    # Total reward calculation with scaled and comprehensive contributions
    total_reward = (
        0.3 * distance_reward +
        0.4 * door_opening_reward -
        0.2 * movement_penalty -
        0.1 * length_penalty
    )

    # Return the total reward and breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': -movement_penalty,
        'length_penalty': -length_penalty
    }
    return total_reward, reward_dict
