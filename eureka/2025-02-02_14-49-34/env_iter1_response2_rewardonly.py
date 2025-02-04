@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, drawer_grasp_pos: torch.Tensor, cabinet_dof_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Distance reward incentivizing agent to move closer to the drawer
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    temp_distance = 1.0
    distance_reward = torch.exp(-temp_distance * distance_to_drawer)
    
    # More nuanced opening reward based on the degree of door opening
    door_opening_progress = (cabinet_dof_pos[:, 3] - 0.0) / (1.0 - 0.0)  # assuming dof_pos goes from 0 to 1 when fully open
    temp_open_revised = 1.0
    open_reward = torch.exp(temp_open_revised * door_opening_progress)

    # Final reward combining both incentives
    total_reward = 0.5 * distance_reward + 0.5 * open_reward

    # Dictionary of reward components
    reward_components = {
        "distance_reward": distance_reward,
        "open_reward": open_reward
    }

    return total_reward, reward_components
