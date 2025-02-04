@torch.jit.script
def compute_reward(velocity: torch.Tensor, heading_proj: torch.Tensor, dof_force: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for transformations
    velocity_temp = 0.1
    heading_temp = 0.1
    force_temp = 0.1
    action_temp = 0.1
    
    # Reward for forward velocity (maximize forward speed)
    forward_velocity_reward = velocity[:, 0]
    forward_velocity_reward = torch.exp(forward_velocity_reward * velocity_temp)

    # Reward for maintaining alignment of the torso
    heading_reward = heading_proj
    heading_reward = torch.exp(heading_reward * heading_temp)

    # Penalty for large forces applied to the joints (energy efficiency)
    force_penalty = torch.sum(dof_force, dim=-1)
    force_penalty = -torch.exp(force_penalty * force_temp)

    # Penalty for large or non-smooth actions
    action_penalty = torch.sum(torch.abs(actions), dim=-1)
    action_penalty = -torch.exp(action_penalty * action_temp)

    # Total reward
    total_reward = forward_velocity_reward + heading_reward + force_penalty + action_penalty

    # Return the total reward and the individual components
    return total_reward, {
        "forward_velocity_reward": forward_velocity_reward,
        "heading_reward": heading_reward,
        "force_penalty": force_penalty,
        "action_penalty": action_penalty
    }
