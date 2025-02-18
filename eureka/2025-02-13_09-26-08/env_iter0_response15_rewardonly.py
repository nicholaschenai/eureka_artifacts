@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor,
                   drawer_grasp_pos: torch.Tensor,
                   cabinet_dof_pos: torch.Tensor,
                   franka_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance the drawer has been moved from its initial position
    drawer_opening_reward = cabinet_dof_pos[:, 3]
    
    # Encourage the agent to move the drawer more effectively
    drawer_opening_temperature = 10.0
    drawer_opening_transformed_reward = torch.exp(drawer_opening_temperature * drawer_opening_reward) - 1.0

    # Encourage minimal movement in the gripper's position to promote efficient movements
    gripper_efficiency_reward = -torch.norm(franka_dof_vel, dim=-1)
    
    # Apply a transformation to scale and stabilize the efficiency reward
    efficiency_temperature = 0.1
    gripper_efficiency_transformed_reward = torch.exp(efficiency_temperature * gripper_efficiency_reward) - 1.0
    
    # Combine the rewards
    total_reward = drawer_opening_transformed_reward + gripper_efficiency_transformed_reward
    
    # Create a reward component dictionary
    rewards_dict = {
        "drawer_opening_reward": drawer_opening_transformed_reward,
        "gripper_efficiency_reward": gripper_efficiency_transformed_reward
    }

    return total_reward, rewards_dict
