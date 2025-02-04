@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Improved Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.5
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature) * 5.0

    # Revised Door Opening Reward
    door_opening_reward_temperature = 5.0
    door_opening_reward_strength = 0.1
    door_opening_reward = door_opening_reward_strength * (torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature) - 1.0)

    # Updated Movement Penalty
    velocity_penalty_factor = 0.001
    movement_penalty = velocity_penalty_factor * torch.abs(cabinet_dof_vel[:, 3])

    # Reinforced Task Completion Reward
    completion_condition = (cabinet_dof_pos[:, 3] > 0.7).float()
    task_completion_reward = completion_condition * 10.0  # Use a high reward to strongly incentivize task completion

    # Total reward calculation
    total_reward = 0.4 * distance_reward + 0.2 * door_opening_reward - 0.1 * movement_penalty + 0.3 * task_completion_reward

    # Return the total reward and individual reward components
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
