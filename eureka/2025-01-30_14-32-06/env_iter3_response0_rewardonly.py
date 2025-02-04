@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate forward velocity towards target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    forward_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_speed = torch.sum(velocity * forward_direction, dim=-1)
    
    # Enhanced Forward Velocity Reward
    forward_vel_temperature = 0.2  # Further increased temperature for even stronger emphasis
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_speed) - 1.0

    # Replaced Upright Reward with a Simpler One
    # Orientation should be naturally maintained for an efficient run.
    reference_up = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    upright_error = torch.sum((up_vec - reference_up) ** 2, dim=-1)  # penalize deviation from upright
    upright_temperature = 0.5
    upright_reward = torch.exp(-upright_temperature * upright_error)

    # Simplified Stability Reward checking accelerations
    ang_velocity = root_states[:, 10:13]
    acceleration = torch.norm(velocity[:, :2], p=2, dim=-1)  # Smooth acceleration
    stability_temperature = 0.1
    stability_reward = torch.exp(-stability_temperature * acceleration)

    # Total reward combines the components
    total_reward = forward_vel_reward + 0.1 * upright_reward + 0.2 * stability_reward

    # Create detailed reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "stability_reward": stability_reward
    }

    return total_reward, reward_dict
