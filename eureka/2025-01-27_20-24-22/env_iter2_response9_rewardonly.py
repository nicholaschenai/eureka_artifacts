@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Enhanced Distance Reward
    distance_temperature = 0.5
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward = torch.exp(-distance_to_handle / distance_temperature)

    # Scaled Door Opening Reward
    door_opening_temperature = 1.0  # Adjusted for reduced impact
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / door_opening_temperature)

    # Redesigned Movement Penalty
    movement_penalty_temperature = 0.1
    movement_penalty = torch.exp(-torch.abs(cabinet_dof_vel[:, 3]) / movement_penalty_temperature)

    # Additional Penalty for prolonged action without outcome
    idle_penalty_factor = 0.1
    idle_penalty = idle_penalty_factor * (1.0 - door_opening_reward)

    # Total reward calculation with weighted contributions
    total_reward = 0.3 * distance_reward + 0.6 * door_opening_reward - 0.1 * movement_penalty - idle_penalty

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'idle_penalty': idle_penalty
    }
    return total_reward, reward_dict
