@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor,
                   drawer_grasp_pos: torch.Tensor,
                   cabinet_dof_pos: torch.Tensor,
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Enhanced Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.2  # Increased sensitivity
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)
    
    # Rescaled Door Opening Reward
    door_opening_reward_temperature = 1.0  # Reduced influence
    door_opening_reward = (torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature) - 1.0) * 0.5

    # Refined Movement Penalty
    movement_penalty_temperature = 0.1
    movement_penalty = torch.exp(-torch.abs(cabinet_dof_vel[:, 3]) / movement_penalty_temperature)

    # Encouraging Task Efficiency
    efficiency_temperature = 0.3
    efficiency_reward = torch.exp(-torch.clamp(cabinet_dof_pos[:, 3], max=1) / efficiency_temperature)

    # Total reward calculation with scaled contributions
    total_reward = (
        0.4 * distance_reward +
        0.3 * door_opening_reward +
        0.1 * movement_penalty +
        0.2 * efficiency_reward
    )

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'efficiency_reward': efficiency_reward
    }
    
    return total_reward, reward_dict
