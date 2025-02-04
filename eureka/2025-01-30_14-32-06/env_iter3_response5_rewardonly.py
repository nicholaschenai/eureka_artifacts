@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, up_vec: torch.Tensor, prev_potentials: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant components from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    # Calculate velocity towards the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_direction = torch.nn.functional.normalize(to_target, dim=-1)
    forward_vel = torch.sum(velocity * target_direction, dim=-1)

    # Forward Velocity Reward (may require further tuning)
    forward_vel_temperature = 0.2
    forward_vel_reward = torch.exp(forward_vel_temperature * forward_vel) - 1.0

    # Simplify Upright Reward - focus on deviations 
    dot_prod_up = torch.sum(up_vec * torch.tensor([0.0, 0.0, 1.0], device=root_states.device).expand_as(up_vec), dim=-1)
    upright_temperature = 0.5
    upright_reward = torch.exp(upright_temperature * (dot_prod_up - 1.0))

    # New Stability Reward - emphasizing minimized angular velocities to encourage smooth movement
    ang_vel_magnitude = torch.norm(ang_velocity, p=2, dim=-1)
    desired_stability = torch.tensor(1.0, device=ang_velocity.device)
    stability_temperature = 0.5
    stability_reward = torch.exp(-stability_temperature * ang_vel_magnitude)

    # Total reward is a weighted sum of the components, re-scaling components for relative importance
    total_reward = 1.0 * forward_vel_reward + 0.1 * upright_reward + 0.15 * stability_reward

    # Construct the reward dictionary
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "stability_reward": stability_reward
    }

    return total_reward, reward_dict
