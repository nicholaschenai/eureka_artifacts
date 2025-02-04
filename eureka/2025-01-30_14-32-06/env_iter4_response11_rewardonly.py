@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, up_vec: torch.Tensor, potentials: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant components from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate velocity towards the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)
  
    # Forward Velocity Reward with new normalization
    forward_vel_temperature = 0.05  # Reduce to control scaling
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Simplify or remove Upright Reward if its impact remains marginal
    # Focusing directly on maintaining a high enough vertical component of the up_vector
    expected_up = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    dot_prod_up = torch.sum(up_vec * expected_up, dim=-1)
    upright_temperature = 0.5  # Increase to better influence
    upright_reward = torch.sigmoid(upright_temperature * (dot_prod_up - 0.9))

    # Stability Reward: continue encouraging smooth motion
    ang_velocity = root_states[:, 10:13]
    stability_temperature = 0.5  # Adjust for broader encouragement
    stability_reward = torch.exp(-stability_temperature * torch.norm(ang_velocity, p=2, dim=-1))

    # Compose the total reward with adjusted emphasis
    total_reward = 1.0 * forward_vel_reward + 0.1 * upright_reward + 0.3 * stability_reward

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "stability_reward": stability_reward
    }

    return total_reward, reward_dict
