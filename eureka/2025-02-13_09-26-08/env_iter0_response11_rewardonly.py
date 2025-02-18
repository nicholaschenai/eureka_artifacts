@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, 
                   franka_lfinger_pos: torch.Tensor, franka_rfinger_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance between the robotic hand and the drawer handle
    to_handle_distance = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    
    # Encourage reducing the distance to the handle
    distance_reward_temp = 0.1
    distance_reward = torch.exp(-distance_reward_temp * to_handle_distance)
    
    # Calculate the closeness of the two fingers
    finger_distance = torch.norm(franka_lfinger_pos - franka_rfinger_pos, dim=-1)
    
    # Encourage the fingers to be close together, indicating a grip
    grip_reward_temp = 0.5
    grip_reward = torch.exp(-grip_reward_temp * finger_distance)
    
    # Reward for the actual opening of the cabinet door
    # Assume the second degree of freedom corresponds to the cabinet door opening
    door_opened_reward = cabinet_dof_pos - 0.5  # Incentivize opening the door by a certain amount
    door_opened_reward = torch.clamp(door_opened_reward, min=0.0)  # Ensure reward is non-negative
    
    # Calculate the overall reward as a weighted sum of the individual components
    total_reward = 1.0 * distance_reward + 1.0 * grip_reward + 3.0 * door_opened_reward
    
    # Compile the individual reward components into a dictionary for monitoring
    reward_components = {
        "distance_reward": distance_reward,
        "grip_reward": grip_reward,
        "door_opened_reward": door_opened_reward
    }
    
    return total_reward, reward_components
