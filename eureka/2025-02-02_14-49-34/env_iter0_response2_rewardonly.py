@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the current drawer position to the target (fully open)
    target_drawer_pos = torch.tensor([0.5, 0.0, 0.0], device=drawer_grasp_pos.device)  # Define target position, adjust as necessary
    distance_to_open = torch.norm(drawer_grasp_pos - target_drawer_pos, dim=-1)
    
    # Calculate reward components
    distance_reward = -distance_to_open
    distance_reward_temperature = 1.0
    distance_reward_transformed = torch.exp(distance_reward / distance_reward_temperature)
    
    # Reward for changing the cabinet DOF position in the positive direction (indicating opening)
    door_opening_reward = cabinet_dof_pos[:, 3]  # Assuming dof[3] corresponds to the door's open angle
    door_opening_reward_temperature = 0.5
    door_opening_reward_transformed = torch.exp(door_opening_reward / door_opening_reward_temperature)

    # Penalize high velocity, encourage directed movement
    velocity_penalty = -torch.norm(cabinet_dof_vel[:, 3:], dim=-1)
    velocity_penalty_temperature = 0.1
    velocity_penalty_transformed = torch.exp(velocity_penalty / velocity_penalty_temperature)
    
    # Combine rewards
    total_reward = distance_reward_transformed + door_opening_reward_transformed + velocity_penalty_transformed

    # Compile individual reward components into a dictionary
    reward_components = {
        'distance_reward': distance_reward_transformed,
        'door_opening_reward': door_opening_reward_transformed,
        'velocity_penalty': velocity_penalty_transformed,
    }
    
    return total_reward, reward_components
