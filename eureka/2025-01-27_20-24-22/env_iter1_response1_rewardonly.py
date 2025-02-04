@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Enhanced reward for minimizing distance to the handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_penalty_temperature = 0.3
    distance_penalty = -torch.exp(-distance_to_handle / distance_penalty_temperature)

    # Re-scaled reward for opening the door
    door_opening_reward_temperature = 5.0
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # Task completion reward (potentially added if the door is sufficiently opened)
    task_completion_threshold = 0.8  # Assuming actuator value needed for "open"
    task_completion_bonus = (cabinet_dof_pos[:, 3] > task_completion_threshold)
    task_completion_reward = torch.where(task_completion_bonus, torch.tensor(1.0, device=cabinet_dof_pos.device), torch.tensor(0.0, device=cabinet_dof_pos.device))

    # Total reward assembly
    total_reward = distance_penalty + 0.5 * door_opening_reward + task_completion_reward

    # Return total and individual components
    reward_dict = {
        'distance_penalty': distance_penalty,
        'door_opening_reward': 0.5 * door_opening_reward,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
