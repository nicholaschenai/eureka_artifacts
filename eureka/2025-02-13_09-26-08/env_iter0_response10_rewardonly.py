@torch.jit.script
def compute_reward(drawer_pos: torch.Tensor, franka_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward variables
    reward_opening = cabinet_dof_pos[:, 3]  # assuming index 3 corresponds to the drawer's degree of freedom for openness
    reward_velocity = cabinet_dof_vel[:, 3]
    
    # Create temperature parameters for transformation
    temperature_opening = 10.0
    temperature_velocity = 5.0
    
    # Transform the rewards
    transformed_opening_reward = 1.0 - torch.exp(-temperature_opening * reward_opening)
    transformed_velocity_reward = torch.exp(temperature_velocity * reward_velocity) - 1.0
    
    # Total reward
    total_reward = transformed_opening_reward + 0.1 * transformed_velocity_reward  # the velocity reward is weighted less

    # Reward components in a dictionary
    reward_components = {
        "opening_reward": transformed_opening_reward,
        "velocity_reward": transformed_velocity_reward
    }
    
    return total_reward, reward_components
