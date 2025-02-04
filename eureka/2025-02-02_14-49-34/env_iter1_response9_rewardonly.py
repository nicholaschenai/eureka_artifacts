@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Adding a stronger incentive for minimizing the distance with an adjusted scale
    temp_distance = 0.5
    distance_reward = 1.0 - torch.tanh(temp_distance * distance_to_drawer)
    
    # Modify the open reward to incentivize the opening of the door more progressively
    target_open_pos = 1.0  # Let's assume the target position for fully opened is 1.0
    door_open_progress = cabinet_dof_pos[:, 3] / target_open_pos
    temp_open = 1.0
    open_reward = torch.tanh(temp_open * door_open_progress)
    
    # Total reward is a combination of being close to the drawer and progressively opening the door
    total_reward = distance_reward + open_reward
    
    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward
    }
    
    return total_reward, reward_components
