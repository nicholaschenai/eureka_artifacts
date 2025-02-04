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
   
    # Forward Velocity Reward with a slightly increased temperature
    forward_vel_temperature = 0.1  # Increased temperature for sharper reward differences
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Upright Reward with redesigned scaling and variability
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    dot_prod_up = torch.sum(up_vec * up_vector_expected, dim=-1)
    # Incorporate more variance by scaling the deviation
    upright_deviation_scale = 2.0
    upright_reward = upright_deviation_scale * (torch.clamp(dot_prod_up, 0.0, 1.0))

    # Balance the total reward more equitably between components
    total_reward = 0.7 * forward_vel_reward + 0.3 * upright_reward

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward
    }

    return total_reward, reward_dict
