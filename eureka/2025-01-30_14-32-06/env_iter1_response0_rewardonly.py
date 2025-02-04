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
    
    # Scaled forward velocity reward
    forward_vel_reward = forward_vel
    
    # Upright reward component (modified for more sensitivity)
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    upright_reward = torch.sum(up_vec * up_vector_expected, dim=-1)

    # Add penalty for excessive deviation from upright
    upright_deviation_penalty = 1.0 - upright_reward
    
    # Normalize and transform the reward components more effectively
    forward_vel_temperature = 0.05  # Less aggressive temperature for fine control
    upright_temperature = 0.2       # Increase temperature for more impact
    deviation_temperature = 0.5     # Stronger penalty for deviation

    forward_vel_reward_transformed = torch.exp(forward_vel_temperature * forward_vel_reward) - 1.0
    upright_reward_transformed = torch.exp(upright_temperature * upright_reward) - 1.0
    deviation_penalty_transformed = -torch.exp(deviation_temperature * upright_deviation_penalty) + 1.0

    # Total reward with adjusted weights and penalty
    reward = forward_vel_reward_transformed + upright_reward_transformed + deviation_penalty_transformed

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "deviation_penalty": upright_deviation_penalty
    }

    return reward, reward_dict
