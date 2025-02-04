@torch.jit.script
def compute_reward(velocity: torch.Tensor, 
                   torso_position: torch.Tensor, 
                   dof_force_tensor: torch.Tensor, 
                   actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Parameters for reward shaping
    velocity_reward_coeff = 1.0
    stability_penalty_coeff = 0.01
    energy_penalty_coeff = 0.001

    # Target forward velocity direction
    forward_velocity = velocity[:, 0]  # Assuming the forward direction is along the x-axis

    # Reward for running fast
    velocity_reward = velocity_reward_coeff * forward_velocity

    # Penalty for using high torques (energy efficiency)
    energy_penalty = energy_penalty_coeff * torch.sum(torch.square(actions), dim=-1)

    # Penalty for deviating from upright position (stability)
    upright_target_height = 1.0  # Set a typical target torso height for upright running
    height_penalty = stability_penalty_coeff * torch.square(torso_position[:, 2] - upright_target_height)

    # Total reward calculation
    total_reward = velocity_reward - energy_penalty - height_penalty

    # Define each component of the reward in a dictionary
    reward_dict = {
        "velocity_reward": velocity_reward,
        "height_penalty": height_penalty,
        "energy_penalty": energy_penalty
    }

    return total_reward, reward_dict
