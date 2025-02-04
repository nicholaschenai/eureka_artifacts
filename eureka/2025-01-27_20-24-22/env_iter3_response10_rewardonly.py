@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Improved Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.2
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Rescaled Door Opening Reward
    door_opening_reward_temperature = 0.9
    door_position = cabinet_dof_pos[:, 3]
    door_opening_reward = 0.1 * torch.tanh(door_position / door_opening_reward_temperature)

    # Refined Movement Penalty
    door_velocity = torch.abs(cabinet_dof_vel[:, 3])
    movement_penalty_temperature = 0.1
    movement_penalty = torch.exp(-door_velocity / movement_penalty_temperature)

    # Re-designed Task Completion Reward
    completion_reward_step = 0.2
    completion_reward = (cabinet_dof_pos[:, 3] > 0.1).float() * completion_reward_step + \
                        (cabinet_dof_pos[:, 3] > 0.3).float() * completion_reward_step + \
                        (cabinet_dof_pos[:, 3] > 0.5).float() * completion_reward_step * 2  # Extra for substantial completion

    # Total reward
    total_reward = (
        0.3 * distance_reward +
        0.3 * door_opening_reward -
        0.2 * movement_penalty +
        0.2 * completion_reward
    )

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'completion_reward': completion_reward
    }
    return total_reward, reward_dict
