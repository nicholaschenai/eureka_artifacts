@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, max_open_deg: float = 90.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Transform the distance reward: Encourage closing the distance
    temp_distance = 1.0
    distance_reward_unscaled = 1.0 - torch.clamp(distance_to_drawer, 0.0, 1.0)
    distance_reward = torch.exp(temp_distance * distance_reward_unscaled)  # Exponential transformation
    
    # Calculate how much the drawer is open
    door_angle_deg = torch.rad2deg(cabinet_dof_pos[:, 3])  # Assume radian inputs

    # Reward the agent as the door opens further
    temp_opening = 0.1
    open_ratio = torch.clamp(door_angle_deg / max_open_deg, 0.0, 1.0)
    open_reward = torch.exp(temp_opening * open_ratio)

    # Introduce a completion bonus for maximum door open angle
    temp_completion = 1.0
    completion_bonus = torch.where(open_ratio > 0.95, torch.tensor(1.0, device=open_reward.device), torch.tensor(0.0, device=open_reward.device))

    # Total reward combines components
    total_reward = 0.3 * distance_reward + 0.5 * open_reward + temp_completion * completion_bonus

    # Adjust open_reward transformation to be usable 
    open_reward = torch.clamp(open_reward, min=0.0, max=1.0)

    # Define reward components
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "completion_bonus": completion_bonus
    }
    
    return total_reward, reward_components
