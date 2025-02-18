@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Improved reward for closing distance to the handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.5  # Adjusted to increase sensitivity 
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Enhance the reward for the extent of opening the drawer
    door_open_value = cabinet_dof_pos[:, 3].clamp(min=0.0)
    opening_reward = 2.0 * torch.clamp(door_open_value, 0, 1.0)  # Linear scaling helping to increase sensitivity

    # Re-scale the velocity reward to ensure balanced contribution
    door_velocity = torch.abs(cabinet_dof_vel[:, 3])
    temperature_velocity = 0.005  # Fine-tuned for smoother reward adjustment
    velocity_reward = torch.exp(-door_velocity / temperature_velocity)

    # Reward composition with new scaling for better balance
    total_reward = 0.3 * dist_reward + 0.5 * opening_reward + 0.2 * velocity_reward

    # Dictionary to track individual reward components
    reward_components = {
        "dist_reward": dist_reward,
        "opening_reward": opening_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
