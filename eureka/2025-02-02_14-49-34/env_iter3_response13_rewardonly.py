@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor, cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Distance penalty: Encourage the hand to be near the drawer
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    temp_distance = 0.5
    distance_reward = 1.0 - torch.tanh(temp_distance * distance_to_drawer)

    # Open reward: Use exponential to reward larger angles, with a higher sensitivity
    door_opening_rad = cabinet_dof_pos[:, 3]
    temp_opening = 2.0
    open_reward = torch.exp(temp_opening * door_opening_rad) - 1.0  # Shifted to ensure non-negative reward
    
    # Optional: Replace speed reward with a stability or safety component to penalize unnecessary acceleration
    # If sticking with velocity, significantly alter its sensitivity
    temp_stability = 0.1
    stability_penalty = -torch.norm(cabinet_dof_vel, dim=-1)  # Penalize high velocity as a stability measure
    stability_penalty = torch.tanh(temp_stability * stability_penalty)

    # Combine the rewards into a total, weighted sum
    total_reward = 0.4 * distance_reward + 0.6 * open_reward + 0.2 * stability_penalty

    # Clamp total reward to ensure it remains non-negative
    total_reward = torch.clamp(total_reward, min=0.0)

    # Creating the reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward,
        "stability_penalty": stability_penalty
    }

    return total_reward, reward_components
