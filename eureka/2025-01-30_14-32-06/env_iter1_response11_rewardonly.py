@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate desired components
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)
    
    # Forward Velocity Reward
    forward_vel_temperature = 0.05   # Slightly reduce temperature for subtler preference
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Upright Reward based on angle deviation
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    dot_product = torch.sum(up_vec * up_vector_expected, dim=-1).clamp(-1.0, 1.0)
    angle_deviation = torch.acos(dot_product)
    upright_reward_temperature = 10.0  # New temperature to amplify uprightness sensitivity

    upright_reward = torch.exp(-upright_reward_temperature * angle_deviation)
    
    # Total reward
    reward = forward_vel_reward + upright_reward

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward
    }

    return reward, reward_dict
