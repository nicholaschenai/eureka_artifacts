@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Calculate velocity reward component
    target_velocity = 10.0  # desired running speed
    velocity_error = target_velocity - torch.norm(velocity[:, 0:2], dim=-1)
    velocity_reward_temp = 0.1
    velocity_reward = torch.exp(-velocity_error**2 / velocity_reward_temp)

    # Calculate potential progress reward to encourage moving towards the target
    progress_reward = potentials - prev_potentials

    # Calculate upright reward component
    upright_temp = 0.5
    upright_reward = torch.exp(-torch.abs(up_vec[:, 2] - 1.0) / upright_temp)

    # Calculate heading alignment reward to encourage running in direction of the target
    heading_temp = 1.0
    heading_reward = torch.exp(-torch.norm(heading_vec - (targets - torso_position)[:, :2], dim=-1) / heading_temp)

    # Total reward as a weighted sum of components
    total_reward = 1.0 * velocity_reward + 1.0 * progress_reward + 0.5 * upright_reward + 0.5 * heading_reward
    
    # Compile reward components into a dictionary
    reward_components = {
        "velocity_reward": velocity_reward,
        "progress_reward": progress_reward,
        "upright_reward": upright_reward,
        "heading_reward": heading_reward
    }

    return total_reward, reward_components
