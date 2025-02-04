@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Distance reward keeps proximity important but reduces weight
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 0.2
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)

    # Opening reward scales more heavily as the drawer opens more
    open_pos_factor = torch.clip(cabinet_dof_pos[:, 3], min=0.0, max=1.0)
    temp_open = 1.0
    open_reward = torch.exp(temp_open * open_pos_factor) - 1.0
    
    # Gradual opening reward instead of a completion bonus
    gradual_open_bonus = torch.tanh(2.0 * open_pos_factor)
    
    # Recalculating weights for better balance
    weight_distance = 0.3
    weight_open = 0.5
    weight_gradual_bonus = 0.2
    total_reward = weight_distance * distance_reward + weight_open * open_reward + weight_gradual_bonus * gradual_open_bonus
    total_reward = torch.clamp(total_reward, min=0.0)

    # Detailed reward breakdown
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "gradual_open_bonus": gradual_open_bonus
    }
    
    return total_reward, reward_components
