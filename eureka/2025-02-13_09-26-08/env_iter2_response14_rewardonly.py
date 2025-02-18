@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward for the robot hand being close to the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.1  # Adjusted for finer scaling importance
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Reward for opening the door; Encourages movement from any starting position
    door_opening = cabinet_dof_pos[:, 3]
    door_opening_reward = door_opening  # Direct positive influence as door is opened

    # Reward for maintaining motion—particularly helpful in ensuring active engagement with the canvas
    velocity = torch.abs(cabinet_dof_vel[:, 3])
    velocity_presence = torch.where(velocity > 0, torch.ones_like(velocity), torch.zeros_like(velocity))
    velocity_reward = velocity_presence * 0.5  # Scaled for better distinguishability

    # Total reward encompassing all dimensions of task progression
    total_reward = 1.0 * dist_reward + 2.0 * door_opening_reward + 0.5 * velocity_reward

    # Structure reward components for analysis
    reward_components = {
        "dist_reward": dist_reward,
        "door_opening_reward": door_opening_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
