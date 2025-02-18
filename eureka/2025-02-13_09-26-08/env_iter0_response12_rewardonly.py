@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor,
                   cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the distance to the target (drawer handle)
    to_target_distance = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Reward for reducing the distance to the target
    distance_temp = 1.0
    distance_reward = torch.exp(-distance_temp * to_target_distance)
    
    # Reward for opening the cabinet door
    # Assume the cabinet_dof_pos[3] represents the door's opening position
    position_temp = 1.0
    opening_reward = torch.exp(position_temp * cabinet_dof_pos[:, 3])  # Encourage opening
    
    # Penalty for large velocities to encourage smooth operation
    velocity_temp = 0.1
    velocity_penalty = -torch.exp(velocity_temp * torch.norm(cabinet_dof_vel[:, 3], dim=-1))
    
    # Calculate the total reward
    total_reward = distance_reward + opening_reward + velocity_penalty
    
    # Return the composed rewards
    rewards_dict = {
        "distance_reward": distance_reward,
        "opening_reward": opening_reward,
        "velocity_penalty": velocity_penalty
    }
    
    return total_reward, rewards_dict
