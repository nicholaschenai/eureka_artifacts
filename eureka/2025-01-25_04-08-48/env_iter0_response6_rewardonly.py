@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant parameters
    velocity = root_states[:, 7:10]  # Linear velocity (assuming x-direction is forward)
    forward_velocity = velocity[:, 0]  # x-component for forward velocity
    
    # Reward for forward velocity
    reward_velocity = forward_velocity
    velocity_temperature = 0.1
    reward_velocity_transformed = torch.exp(velocity_temperature * reward_velocity) - 1.0

    # Penalize for high joint forces/torques to encourage efficient movement
    force_penalty = -torch.sum(torch.abs(dof_force), dim=-1) * dt  
    force_temperature = 0.01
    force_penalty_transformed = torch.exp(force_temperature * force_penalty) - 1.0

    # Stability bonus might be based on a simple condition like being within certain vertical range

    # Total reward
    total_reward = reward_velocity_transformed + force_penalty_transformed

    # Reward components dictionary
    reward_components = {
        "reward_velocity": reward_velocity_transformed,
        "force_penalty": force_penalty_transformed
    }

    return total_reward, reward_components
