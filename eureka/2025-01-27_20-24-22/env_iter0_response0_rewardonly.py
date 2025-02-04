@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Distance reward: Negative of the distance between franka's grasp position and the drawer's grasp position
    distance_to_drawer = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    distance_reward = -distance_to_drawer
    
    # Threshold for considering the drawer as "open"
    drawer_open_threshold: float = 0.2
    
    # Drawer movement reward: Positive reward if the drawer's DOF position has exceeded the threshold for being "open"
    drawer_open_reward = (cabinet_dof_pos[:, 3] > drawer_open_threshold).float()
    
    # Temperature parameters for transformation
    distance_temperature: float = 0.1
    open_temperature: float = 0.1
    
    # Apply transformations to normalize reward components
    transformed_distance_reward = torch.exp(distance_temperature * distance_reward)
    transformed_open_reward = torch.exp(open_temperature * drawer_open_reward)
    
    # Total reward as a weighted sum of the components
    total_reward = 0.8 * transformed_distance_reward + 0.2 * transformed_open_reward
    
    # Package individual components for debugging/analysis
    reward_components: Dict[str, torch.Tensor] = {
        "distance_reward": transformed_distance_reward,
        "open_reward": transformed_open_reward
    }
    
    return total_reward, reward_components
