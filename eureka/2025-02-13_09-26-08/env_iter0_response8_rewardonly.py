@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, 
                   franka_dof_vel: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the distance between the robot hand and the drawer handle
    distance_to_drawer = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    
    # Compute velocity of the drawer motion
    drawer_velocity = cabinet_dof_pos[:, 3]
    
    # Define temperature parameters for transformations
    distance_temp = 1.0
    velocity_temp = 0.1
    
    # Compute reward components
    distance_reward = -torch.exp(-distance_temp * distance_to_drawer)
    velocity_reward = torch.exp(velocity_temp * drawer_velocity)
    
    # Total reward is a combination of getting closer to the drawer and moving it
    total_reward = distance_reward + velocity_reward
    
    # Return total reward and individual components
    return total_reward, {
        "distance_reward": distance_reward,
        "velocity_reward": velocity_reward
    }
