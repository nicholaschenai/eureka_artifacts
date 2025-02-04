@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for transformations
    temp_distance = 1.0
    temp_opening = 0.3
    
    # Distance to the drawer
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Inverted and normalized distance reward
    distance_reward = 1.0 - torch.tanh(temp_distance * distance_to_drawer)
   
    # Reward for opening the cabinet door
    door_opening_rad = cabinet_dof_pos[:, 3]
    opening_reward = torch.sigmoid(temp_opening * door_opening_rad)
    
    # The speed reward component is discarded due to non-effectiveness

    # Weighting of each component
    weight_distance = 0.4
    weight_opening = 0.6

    # Compute the total reward
    total_reward = weight_distance * distance_reward + weight_opening * opening_reward

    # Clamp total reward to ensure it remains non-negative
    total_reward = torch.clamp(total_reward, min=0.0)

    # Dictionary of individual reward components
    reward_components = {
        "distance_reward": distance_reward,
        "opening_reward": opening_reward,
    }
    
    return total_reward, reward_components
