@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor,
                   drawer_open_pos: torch.Tensor,
                   franka_dof_vel: torch.Tensor,
                   cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Distance between the robot's gripper and the drawer handle
    dist_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=1)
    
    # Reward for reducing the distance to the drawer handle
    distance_reward = -dist_to_handle
    distance_temp = 1.0
    scaled_distance_reward = torch.exp(-distance_temp * distance_reward)

    # Measure how far the drawer has been opened
    drawer_opening_reward = cabinet_dof_pos[:, 3]
    opening_temp = 1.0
    scaled_opening_reward = torch.exp(opening_temp * drawer_opening_reward)

    # Penalize large robot arm velocities
    velocity_penalty = torch.sum(franka_dof_vel * franka_dof_vel, dim=1)
    vel_penalty_temp = 0.1
    scaled_velocity_penalty = torch.exp(-vel_penalty_temp * velocity_penalty)

    # Total reward
    total_reward = scaled_distance_reward + scaled_opening_reward - scaled_velocity_penalty
    
    reward_components = {
        "distance_reward": scaled_distance_reward,
        "opening_reward": scaled_opening_reward,
        "velocity_penalty": -scaled_velocity_penalty
    }
    
    return total_reward, reward_components
