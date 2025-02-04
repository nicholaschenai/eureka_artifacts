@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract components from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate velocity towards the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)
    
    # Adjusted Forward Velocity Reward
    forward_vel_temperature = 0.1  # Increased temperature to amplify positive forward movements
    forward_vel_reward = torch.atan(forward_vel_temperature * forward_vel)
    
    # Revised Upright Reward component
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    dot_prod_up = torch.sum(up_vec * up_vector_expected, dim=-1)
    # Applying a square transformation to emphasize deviations
    upright_temperature = 0.2  # Reduced temperature but altered transformation
    upright_reward = torch.max(torch.zeros_like(dot_prod_up), torch.square(upright_temperature * (dot_prod_up - 1.0)))

    # Total reward is a weighted sum of the components
    total_reward = 2.0 * forward_vel_reward + upright_reward

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward
    }

    return total_reward, reward_dict
