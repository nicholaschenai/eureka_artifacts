@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso velocity from the root state
    velocity = root_states[:, 7:10]  # Assuming [7:10] indices capture the velocity
    forward_velocity = velocity[:, 0]  # We focus on the x-axis for forward running

    # Reward component for forward velocity
    velocity_reward = forward_velocity

    # Penalty component for action usage to prevent excessive energy use
    action_penalty_weight: float = 0.01  # This weight can be tuned
    action_penalty = -action_penalty_weight * torch.norm(actions, p=2, dim=-1)

    # Combine the reward components
    total_reward = velocity_reward + action_penalty

    # To bound the velocity reward, transform it
    velocity_temperature: float = 1.0  # Specify temperature
    velocity_reward_transformed = torch.exp(velocity_reward / velocity_temperature) - 1.0

    # Total transformed reward
    total_transformed_reward = velocity_reward_transformed + action_penalty

    # Prepare reward components dictionary
    reward_components = {
        "velocity_reward": velocity_reward,
        "action_penalty": action_penalty,
        "total_transformed_reward": total_transformed_reward
    }

    return total_transformed_reward, reward_components
