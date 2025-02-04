@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Improved Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.5
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Rescaled Door Opening Reward
    door_opening_reward_temperature = 1.0
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # Penalty for inefficient distance
    distance_penalty_temperature = 0.5
    distance_penalty = torch.exp(distance_to_handle * cabinet_dof_vel[:, 3] / distance_penalty_temperature)

    # Success Bonus for task completion
    success_threshold = 0.75  # Assuming full door opening is at 1.0
    success_bonus = torch.where(cabinet_dof_pos[:, 3] > success_threshold, torch.tensor(10.0, device=franka_grasp_pos.device), torch.tensor(0.0, device=franka_grasp_pos.device))

    # Total reward calculation with adjusted contributions
    total_reward = distance_reward - 0.2 * distance_penalty + 0.5 * door_opening_reward + success_bonus

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'distance_penalty': distance_penalty,
        'success_bonus': success_bonus
    }
    return total_reward, reward_dict
