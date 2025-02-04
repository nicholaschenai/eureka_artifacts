@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Reward the agent for minimizing the distance to the drawer
    # Adjust the temperature to make the scaling more effective
    temp_distance = 2.0
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)
    
    # Reward for the actual amount the door is opened
    # Apply a temperature scaling for a smoother gradient
    temp_open = 10.0
    open_reward = torch.sigmoid(temp_open * (cabinet_dof_pos[:, 3] - 0.5))  # assuming full open is 1, scaled midpoint is 0.5

    # Total reward is a combination of being close to the drawer and having the door opened progressively
    total_reward = distance_reward + open_reward
    
    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward
    }
    
    return total_reward, reward_components
