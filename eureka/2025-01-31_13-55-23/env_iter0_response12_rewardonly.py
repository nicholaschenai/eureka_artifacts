@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]  # assuming x-axis is the forward direction

    # Reward for moving forward
    forward_reward = forward_velocity

    # Penalty for excessive actions (encourages smoother, energy-efficient control)
    energy_penalty = torch.sum(actions ** 2, dim=-1)

    # Reward for being upright (up vector should be aligned with positive z-axis)
    upright_reward = up_vec[:, 2]  # assuming the z-up axis

    # Parameters for shaping rewards
    forward_temp = 1.0  
    energy_temp = -0.01  
    upright_temp = 1.0  

    # Applying transformations
    transformed_forward_reward = torch.exp(forward_reward / forward_temp)
    transformed_energy_penalty = torch.exp(energy_penalty / energy_temp)  # negative penalty when multiplied
    transformed_upright_reward = torch.exp(upright_reward / upright_temp)

    # Total reward construction
    reward = transformed_forward_reward + transformed_upright_reward - transformed_energy_penalty

    # Individual reward components for debugging or analysis
    reward_components = {
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_energy_penalty": transformed_energy_penalty,
        "transformed_upright_reward": transformed_upright_reward
    }

    return reward, reward_components
