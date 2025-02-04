@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Improved Distance Reward: Enhance sensitivity
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.15
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Adjusted Door Opening Reward: Proper scaling
    door_angle_progress = torch.clamp(cabinet_dof_pos[:, 3], min=0.0, max=1.0)
    door_opening_reward_temperature = 0.20
    door_opening_reward = torch.exp(door_angle_progress / door_opening_reward_temperature) - 1.0

    # Refined Movement Penalty: Balance penalty influence
    door_velocity = cabinet_dof_vel[:, 3]
    movement_penalty_temperature = 0.25
    movement_penalty = torch.exp(-torch.abs(door_velocity) / movement_penalty_temperature)

    # Adjusted Task Completion Reward: Trigger when significant progress is made
    task_completion_threshold = 0.7  # Activating earlier
    task_completion = (cabinet_dof_pos[:, 3] > task_completion_threshold).float()
    task_completion_reward = task_completion * 2.5  # Highlight task success once a threshold is achieved

    # Total reward with balanced contribution
    total_reward = 0.3 * distance_reward + 0.2 * door_opening_reward - 0.2 * movement_penalty + 0.3 * task_completion_reward

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
