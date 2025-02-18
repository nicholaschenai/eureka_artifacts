@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor,
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Reward component for minimizing the distance between the robot's hand and the drawer handle
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    dist_reward_temp = 10.0  # temperature parameter for distance reward
    distance_reward = torch.exp(-dist_reward_temp * distance_to_drawer)

    # Reward component for maximizing the opening angle of the cabinet drawer
    opening_angle = cabinet_dof_pos[:, 3]  # extract the relevant DOF position for the drawer
    opening_reward_temp = 1.0  # temperature parameter for opening reward
    opening_reward = torch.exp(opening_reward_temp * opening_angle)

    # Reward component for minimizing the velocity of the drawer (for smoother operation)
    drawer_velocity = cabinet_dof_vel[:, 3]  # extract the relevant DOF velocity for the drawer
    velocity_reward_temp = 1.0  # temperature parameter for velocity reward
    velocity_reward = torch.exp(-velocity_reward_temp * torch.abs(drawer_velocity))

    # Combine all rewards
    total_reward = distance_reward + opening_reward + velocity_reward

    # Create a reward component dictionary for analysis
    reward_dict = {
        "distance_reward": distance_reward,
        "opening_reward": opening_reward,
        "velocity_reward": velocity_reward
    }
    
    return total_reward, reward_dict
