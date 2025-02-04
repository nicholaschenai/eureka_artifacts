@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract components from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]
    
    # Calculate velocity towards the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)

    # Increase influence on Forward Velocity Reward
    forward_vel_temperature = 0.1  # Increased temperature for more aggressive encouragement
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Re-write Upright Reward to ensure variance and adaptability
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    dot_prod_up = torch.sum(up_vec * up_vector_expected, dim=-1)
    upright_temperature = 0.2  # Reduced temperature
    upright_reward = torch.exp(upright_temperature * (dot_prod_up - 0.95))  # Slightly lower baseline

    # Add Angular Velocity Punishment
    ang_vel_penalty_temperature = 0.1
    ang_vel_magnitude = torch.norm(ang_velocity, p=2, dim=-1)
    ang_vel_penalty = torch.exp(-ang_vel_penalty_temperature * ang_vel_magnitude)  # Penalize high angular velocity

    # Total reward is a weighted sum of the components
    total_reward = 1.5 * forward_vel_reward + 0.2 * upright_reward + 0.3 * ang_vel_penalty

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "ang_vel_penalty": ang_vel_penalty
    }

    return total_reward, reward_dict
