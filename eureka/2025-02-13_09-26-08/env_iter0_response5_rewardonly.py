@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for transformations
    distance_temperature = 10.0
    dof_movement_temperature = 1.0

    # Calculate distance between the gripper and the drawer grasp position
    distance_reward = -torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    
    # Exponential transformation for distance reward normalized using temperature
    transformed_distance_reward = torch.exp(distance_temperature * distance_reward)

    # Reward based on opening the cabinet (we assume that a higher dof position indicates more opening)
    # Give a positive reward for moving the cabinet door to a more open position
    dof_movement_reward = cabinet_dof_pos[:, 3]
    
    # Exponential transformation for dof movement reward normalized using temperature
    transformed_dof_movement_reward = torch.exp(dof_movement_temperature * dof_movement_reward)

    # Total reward combines both components, assigning more weight to opening the cabinet
    total_reward = 0.5 * transformed_distance_reward + 2.0 * transformed_dof_movement_reward

    # Return the total reward and a breakdown of each component
    reward_components = {
        'distance_reward': transformed_distance_reward,
        'dof_movement_reward': transformed_dof_movement_reward
    }

    return total_reward, reward_components
