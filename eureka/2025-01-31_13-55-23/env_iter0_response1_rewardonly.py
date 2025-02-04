@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant information from the root_states
    velocity = root_states[:, 7:10]  # Extract linear velocity
    ang_velocity = root_states[:, 10:13]  # Extract angular velocity

    # Forward velocity is the velocity on the x-axis (assuming the x-axis is the forward direction)
    forward_velocity = velocity[:, 0]  # Consider the x-component as forward

    # Reward for high forward velocity
    velocity_reward = forward_velocity

    # Penalize unnecessary angular velocities (yaw, pitch, roll rates)
    angular_penalty_weight = 0.1
    angular_penalty = angular_penalty_weight * torch.norm(ang_velocity, p=2, dim=-1)

    # Total reward
    total_reward = velocity_reward - angular_penalty

    # Return the total reward and a breakdown of the components
    reward_components = {
        "velocity_reward": velocity_reward,
        "angular_penalty": angular_penalty
    }
    return total_reward, reward_components
