@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the distance from the hand's grasp position to the drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)

    # Redefine the distance reward: more critical transformation to be more encouraging towards minimization
    temp_distance = 2.0  # Heavier penalty for distance
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)  # Exponential encouraging closeness

    # Open reward: emphasize the progression in opening the door
    door_opening_progress = cabinet_dof_pos[:, 3]  # Directly use progress, assume correct normalization
    temp_opening = 1.0
    open_reward = torch.sigmoid(temp_opening * (door_opening_progress - 0.5))  # Transform to focus on opening progress

    # Total reward is a normalized combination of the two components
    weight_distance = 0.4  # Shift focus towards door opening
    weight_open = 0.6
    total_reward = weight_distance * distance_reward + weight_open * open_reward

    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward
    }
    
    return total_reward, reward_components
