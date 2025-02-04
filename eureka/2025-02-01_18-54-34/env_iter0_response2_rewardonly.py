@torch.jit.script
def compute_reward(velocity: torch.Tensor, heading_vec: torch.Tensor, up_vec: torch.Tensor, potentials: torch.Tensor, 
                   targets: torch.Tensor, torso_position: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Parameters for reward shaping
    speed_coeff = 1.0
    heading_coeff = 0.5
    upright_coeff = 0.2
    potentials_coeff = 0.3
    
    # Compute velocity towards the target
    to_target_vector = targets - torso_position
    to_target_unit = torch.nn.functional.normalize(to_target_vector, p=2, dim=-1)
    forward_velocity = torch.sum(velocity * to_target_unit, dim=-1)
    
    # Reward component for speed
    speed_reward = forward_velocity * speed_coeff

    # Heading alignment reward
    heading_alignment = torch.sum(heading_vec * to_target_unit, dim=-1)
    heading_reward = heading_coeff * heading_alignment

    # Uprightness reward (encourages the humanoid to stay upright)
    uprightness = up_vec[..., 2]  # z-component of the up vector
    upright_reward = upright_coeff * uprightness

    # Potential-based reward (to encourage reaching target)
    potential_reward = potentials * potentials_coeff

    # Total reward
    total_reward = speed_reward + heading_reward + upright_reward + potential_reward

    # Transformation to bound the reward range
    speed_temp = 0.1
    heading_temp = 0.1
    upright_temp = 0.1
    potentials_temp = 0.1

    transformed_speed_reward = torch.exp(speed_reward / speed_temp) - 1.0
    transformed_heading_reward = torch.exp(heading_reward / heading_temp) - 1.0
    transformed_upright_reward = torch.exp(upright_reward / upright_temp) - 1.0
    transformed_potential_reward = torch.exp(potential_reward / potentials_temp) - 1.0

    # Final total reward with transformations
    final_reward = transformed_speed_reward + transformed_heading_reward + transformed_upright_reward + transformed_potential_reward

    # Reward components for debugging and analysis
    reward_dict = {
        "speed_reward": speed_reward,
        "heading_reward": heading_reward,
        "upright_reward": upright_reward,
        "potential_reward": potential_reward
    }

    return final_reward, reward_dict
