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
    
    # Adjusted Reward for forward velocity
    forward_vel_temperature = 0.2
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Revised Upright reward component to be more challenging
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    dot_prod_up = torch.sum(up_vec * up_vector_expected, dim=-1)
    upright_temperature = 0.1  # Reduced temperature for fine-tuning
    upright_reward = dot_prod_up

    # Increased Stability Reward to encourage more steady movements
    ang_velocity = root_states[:, 10:13]
    stability_temperature = 0.4
    stability_reward = torch.exp(-stability_temperature * torch.norm(ang_velocity, p=2, dim=-1))

    # Total reward is a weighted sum of the components with adjusted weights
    total_reward = 2.0 * forward_vel_reward + 0.2 * upright_reward + 1.0 * stability_reward

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "stability_reward": stability_reward
    }

    return total_reward, reward_dict
