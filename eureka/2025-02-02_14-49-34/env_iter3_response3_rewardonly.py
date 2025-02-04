@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Re-evaluating distance reward with more sensitivity
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 2.0  # Increase sensitivity
    distance_reward = torch.exp(-temp_distance * distance_to_drawer).clamp(max=1.0)
    
    # Adjusting open reward for consistency
    temp_opening = 0.05
    open_reward = torch.sigmoid(temp_opening * torch.rad2deg(cabinet_dof_pos[:, 3]))
    
    # Redefining speed reward to provide effective guidance
    temp_velocity = 5.0  # Increase scaling to make positive feedback more distinguishable
    speed_reward = torch.sigmoid(temp_velocity * cabinet_dof_vel[:, 3].clamp(min=0.0))
    
    # Introducing a task completion component
    temp_completion = 0.01
    completion_reward = torch.sigmoid(temp_completion * (cabinet_dof_pos[:, 3] - torch.pi / 2).abs())
    
    # Weighting the components
    weight_distance = 0.2
    weight_open = 0.4
    weight_speed = 0.2
    weight_completion = 0.2
    total_reward = (
        weight_distance * distance_reward
        + weight_open * open_reward
        + weight_speed * speed_reward
        + weight_completion * completion_reward
    )

    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "speed_reward": speed_reward,
        "completion_reward": completion_reward
    }
    
    return total_reward, reward_components
