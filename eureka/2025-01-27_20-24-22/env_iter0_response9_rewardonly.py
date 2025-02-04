@torch.jit.script
def compute_reward(drawer_pos: torch.Tensor, franka_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Calculate the distance to target
    distance_to_target = torch.norm(drawer_pos - franka_grasp_pos, p=2, dim=-1)
    distance_reward = -distance_to_target  # Negative reward for distance to target

    # Reward for opening the drawer based on DOF position (assuming larger values indicate more open)
    door_opening_reward = cabinet_dof_pos[:, 3]
    
    # Velocity reward encouraging movement in the opening direction
    velocity_reward_temperature = 1.0
    velocity_reward = torch.exp(velocity_reward_temperature * cabinet_dof_vel[:, 3]) - 1.0
    
    # Total reward combining components
    total_reward = distance_reward + door_opening_reward + velocity_reward

    # Return total reward and individual components
    reward_components = {
        "distance_reward": distance_reward,
        "door_opening_reward": door_opening_reward,
        "velocity_reward": velocity_reward
    }
    
    return total_reward, reward_components
