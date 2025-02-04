@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor, ang_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant components from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate velocity towards the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)
    
    # Reward for forward velocity
    forward_vel_reward = forward_vel
    
    # Upright reward component
    up_vector_expected = torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec)
    upright_reward = torch.sum(up_vec * up_vector_expected, dim=-1)

    # Penalty for angular velocity
    ang_vel_penalty = torch.sum(torch.abs(ang_velocity), dim=-1)

    # Normalizing and transforming the reward components
    forward_vel_temperature = 0.05  # Lower temperature for more scaled forward velocity
    upright_temperature = 0.2       # Higher temperature to give more emphasis on uprightness
    ang_vel_penalty_weight = 0.05   # Weight for angular velocity penalty

    forward_vel_reward_transformed = torch.exp(forward_vel_temperature * forward_vel_reward) - 1.0
    upright_reward_transformed = torch.exp(upright_temperature * upright_reward) - 1.0
    ang_vel_penalty_transformed = ang_vel_penalty_weight * ang_vel_penalty

    # Total reward
    reward = forward_vel_reward_transformed + upright_reward_transformed - ang_vel_penalty_transformed

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward_transformed,
        "upright_reward": upright_reward_transformed,
        "ang_vel_penalty": -ang_vel_penalty_transformed  # to indicate this is a penalty
    }

    return reward, reward_dict
