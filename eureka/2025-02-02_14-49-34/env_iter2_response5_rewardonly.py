@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    to_drawer_dist = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Distance reward with improved scaling
    temp_distance = 5.0
    distance_reward = torch.exp(-temp_distance * to_drawer_dist)  # Transform to provide exponential reward closer to the target
    
    # Improved open reward: increasing with how much the door is opened and re-scaling
    door_angle = cabinet_dof_pos[:, 3]
    max_angle = torch.tensor(1.57, device=door_angle.device)  # 90 degrees in radian as max open angle
    temp_opening = 1.0
    open_reward = torch.clamp(door_angle / max_angle, 0.0, 1.0)  # Normalize by max angle and clamp
    
    # Reward for reducing episode length (encourages quicker completion)
    temp_length = 0.002
    length_reward = -temp_length * (500.0 - door_angle)  # Encourage quick open

    # Combine rewards with updated weights
    weight_distance = 0.3
    weight_open = 0.6
    weight_length = 0.1
    total_reward = (weight_distance * distance_reward + 
                    weight_open * open_reward +
                    weight_length * length_reward)

    # Creating the dictionary for individual reward components
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "length_reward": length_reward
    }

    return total_reward, reward_components
