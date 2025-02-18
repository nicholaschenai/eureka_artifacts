@torch.jit.script
def compute_reward(drawer_grasp_pos: torch.Tensor, franka_grasp_pos: torch.Tensor,
                   cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the distance from the Franka's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Encourage reduction in distance to the drawer
    distance_reward_gain = 1.0
    distance_reward = -distance_to_drawer * distance_reward_gain
    
    # Reward for the cabinet door's opening; assume target opening angle is at maximum limit
    target_cabinet_open_angle = torch.pi / 2  # assuming 90 degrees is the fully open position
    open_pos_reward_gain = 2.0
    open_pos_reward = -torch.abs(cabinet_dof_pos - target_cabinet_open_angle) * open_pos_reward_gain
    
    # Encourage velocity towards the target opening
    velocity_reward_gain = 0.5
    velocity_reward = torch.clamp(cabinet_dof_vel, max=0) * velocity_reward_gain
    
    # Total reward
    total_reward = distance_reward + open_pos_reward + velocity_reward

    # Temperature parameters for transformations
    distance_temp = 0.1
    open_pos_temp = 0.1
    velocity_temp = 0.1

    # Apply exponential transformations to individual components
    distance_reward = torch.exp(distance_reward / distance_temp)
    open_pos_reward = torch.exp(open_pos_reward / open_pos_temp)
    velocity_reward = torch.exp(velocity_reward / velocity_temp)

    # Regenerate total reward after transformations for better reward scaling
    total_reward = distance_reward + open_pos_reward + velocity_reward
    
    # Construct the component reward dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_pos_reward": open_pos_reward,
        "velocity_reward": velocity_reward
    }
    
    return total_reward, reward_components
