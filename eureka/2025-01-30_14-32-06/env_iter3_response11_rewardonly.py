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

    # Enhance Forward Velocity Reward
    forward_vel_temperature = 0.1
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Modified Upright Reward, focusing only on penalties
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    dot_prod_up = torch.sum(up_vec * up_vector_expected, dim=-1)
    upright_threshold = 0.8
    upright_penalty_temperature = 0.5
    upright_penalty = torch.where(dot_prod_up < upright_threshold, torch.exp(-upright_penalty_temperature * (dot_prod_up - upright_threshold)), torch.tensor(0.0, device=root_states.device))

    # Energy Efficiency Penalty to replace stability
    ang_velocity = root_states[:, 10:13]
    energy_penalty_temperature = 0.05
    energy_penalty = torch.exp(energy_penalty_temperature * torch.norm(ang_velocity, p=2, dim=-1)) - 1.0
    
    # Total reward as a weighted sum of components
    total_reward = 1.5 * forward_vel_reward - 0.1 * upright_penalty - 0.05 * energy_penalty

    # Construct the reward component dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_penalty": upright_penalty,
        "energy_penalty": energy_penalty
    }

    return total_reward, reward_dict
