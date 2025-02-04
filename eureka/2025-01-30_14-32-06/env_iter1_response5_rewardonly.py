@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant components from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate velocity towards the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)
    
    # Apply an adjusted scale to the forward velocity component
    forward_vel_temperature = 0.05  # Reduced temperature for smoother scaling
    forward_vel_reward_transformed = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Upright reward component with higher sensitivity
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    upright_reward = torch.sum(up_vec * up_vector_expected, dim=-1)
    upright_temperature = 0.5  # Increased temperature for better adjustment
    upright_reward_transformed = torch.exp(upright_temperature * upright_reward) - 1.0

    # Add a slight penalty for excessive angular movement to improve stability
    ang_velocity = root_states[:, 10:13]
    angular_penalty_coeff = 0.01
    angular_penalty = angular_penalty_coeff * torch.sum(torch.abs(ang_velocity), dim=-1)

    # Total reward
    reward = forward_vel_reward_transformed + upright_reward_transformed - angular_penalty

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward_transformed,
        "upright_reward": upright_reward_transformed,
        "angular_penalty": -angular_penalty
    }

    return reward, reward_dict
