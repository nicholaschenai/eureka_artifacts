@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor,
                   drawer_grasp_pos: torch.Tensor,
                   cabinet_dof_pos: torch.Tensor,
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Refined Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.15
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Rescaled Door Opening Reward
    door_opening_reward_temperature = 0.5
    rescaled_door_opening = torch.tanh(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # Discard Movement Penalty as it doesn't contribute effectively
    movement_penalty = torch.tensor(0.0, device=cabinet_dof_pos.device)

    # Reframed Task Completion Reward
    task_completion_threshold = 0.5
    task_completion = torch.where(cabinet_dof_pos[:, 3] > task_completion_threshold, 1.0, 0.0)
    task_completion_reward = task_completion * 5.0  # Increased incentive for threshold achievement

    # Total reward
    total_reward = 0.4 * distance_reward + 0.4 * rescaled_door_opening + 0.2 * task_completion_reward

    # Return total reward and breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': rescaled_door_opening,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }

    return total_reward, reward_dict
