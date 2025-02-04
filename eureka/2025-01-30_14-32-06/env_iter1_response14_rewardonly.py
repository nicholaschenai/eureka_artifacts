@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant components from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate velocity towards the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)
    
    # Adjusted Forward Velocity Reward with a new transformation
    forward_vel_temperature = 0.2  # Increased temperature for more sensitivity
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Enhanced Upright Reward
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    upright_reward_value = torch.sum(up_vec * up_vector_expected, dim=-1)
    # Introduce non-linear transformation with higher sensitivity
    upright_temperature = 0.5
    upright_reward = torch.exp(upright_temperature * (upright_reward_value - 0.9)) - 1.0 

    # Total reward
    reward = forward_vel_reward + upright_reward

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward
    }

    return reward, reward_dict
