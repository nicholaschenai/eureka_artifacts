@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the Euclidean distance to the drawer
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Scale down and represent the distance penalty, encouraging the agent to reduce the distance
    temp_distance = 0.5
    distance_reward = 1.0 / (1.0 + temp_distance * distance_to_drawer)

    # Encourage the opening of the door, apply sigmoid for gradual opening reward
    open_distance = cabinet_dof_pos[:, 3]
    open_threshold = 0.1
    open_reward = 1 / (1 + torch.exp(-10 * (open_distance - open_threshold)))

    # Introduce a small penalty for spurious movements to encourage efficient opening
    movement_penalty = torch.abs(cabinet_dof_vel[:, 3])
    temp_movement = 0.01
    combined_movement_penalty = temp_movement * movement_penalty

    # The total reward combines all aspects with proper weighting
    total_reward = distance_reward + open_reward - combined_movement_penalty

    # Construct the reward components dictionary for analysis
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "movement_penalty": combined_movement_penalty
    }

    return total_reward, reward_components
