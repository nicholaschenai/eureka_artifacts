@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, up_vec: torch.Tensor, velocity: torch.Tensor, ang_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant components from root_states
    torso_position = root_states[:, 0:3]
    
    # Calculate velocity towards the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)

    # Adjusted Forward Velocity Reward
    forward_vel_temperature = 0.1
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Upright Reward component re-scaled and transformed
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    dot_prod_up = torch.sum(up_vec * up_vector_expected, dim=-1)
    upright_temperature = 1.0
    upright_reward = torch.exp(upright_temperature * (dot_prod_up - 1.0))

    # Angular Velocity Penalty
    ang_vel_magnitude = torch.norm(ang_velocity, p=2, dim=-1)
    ang_velocity_penalty_temperature = 0.05
    ang_velocity_penalty = -torch.exp(ang_velocity_penalty_temperature * ang_vel_magnitude)

    # Total reward is a weighted sum of the components
    total_reward = 1.0 * forward_vel_reward + 0.3 * upright_reward + 0.2 * ang_velocity_penalty

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "angular_velocity_penalty": ang_velocity_penalty
    }

    return total_reward, reward_dict
