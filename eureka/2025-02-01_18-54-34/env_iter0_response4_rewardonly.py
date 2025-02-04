@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity of the humanoid
    forward_velocity = root_states[:, 7]  # Assuming Y-axis is the forward direction

    # Reward for forward velocity
    velocity_reward_temperature = 0.1
    velocity_reward = torch.exp(velocity_reward_temperature * forward_velocity)

    # Penalize large joint forces to encourage energy efficiency
    force_penalty_temperature = -0.01
    force_penalty = torch.exp(force_penalty_temperature * torch.norm(dof_force_tensor, p=2, dim=-1))

    # Penalize extreme actions to promote smooth motion
    action_penalty_temperature = -0.01
    action_penalty = torch.exp(action_penalty_temperature * torch.norm(actions, p=2, dim=-1))

    # Total reward combines all components
    total_reward = velocity_reward + force_penalty + action_penalty

    reward_dict = {
        "velocity_reward": velocity_reward,
        "force_penalty": force_penalty,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_dict
