@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Reward the agent for minimizing the distance to the drawer
    temp_distance = 2.0
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)  # Providing a steeper gradient for close proximity

    # Redefine the open reward (use exponential transformation for continuous reward scaling)
    temp_opening = 0.1
    open_reward = torch.exp(temp_opening * (cabinet_dof_pos[:, 3])) - 1  # Encourages further opening, properly rescaled

    # Include a bonus for reaching a certain threshold of openness as additional incentive
    threshold_open = torch.deg2rad(45.0)  # Threshold angle in radians
    openness_bonus = torch.where(cabinet_dof_pos[:, 3] > threshold_open, torch.tensor(1.0, device=cabinet_dof_pos.device), torch.tensor(0.0, device=cabinet_dof_pos.device))

    # Total reward weights
    weight_distance = 0.4
    weight_open = 0.5
    weight_bonus = 0.1

    # Calculate the total reward
    total_reward = (weight_distance * distance_reward) + (weight_open * open_reward) + (weight_bonus * openness_bonus)
    
    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "openness_bonus": openness_bonus
    }
    
    return total_reward, reward_components
