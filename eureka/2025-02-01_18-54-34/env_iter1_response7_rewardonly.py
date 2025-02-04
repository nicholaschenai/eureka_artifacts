@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocities
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Modify forward reward with a higher constant for max speed
    max_speed = 12.0
    forward_reward = forward_velocity / max_speed

    # Increase the weight of the forward reward
    forward_reward_weight = 1.5

    # Penalize sideways and backward movement
    sideways_velocity = torch.norm(velocity[:, 1:3], p=2, dim=-1)
    sideways_penalty = -sideways_velocity / (max_speed / 2.0)

    # Increase sideways penalty scale
    sideways_penalty_weight = 1.2

    # Modify heading reward component
    # Assuming heading is perfectly aligned is incorrect. Use a designed heading difference
    heading_difference = torch.abs(forward_velocity) - torch.abs(sideways_velocity)
    heading_reward = torch.clamp(heading_difference, 0.0, 1.0)

    # Combine transformed rewards
    temperature1 = 0.7
    temperature2 = 0.6
    transformed_forward_reward = torch.exp(temperature1 * forward_reward) - 1.0
    transformed_sideways_penalty = torch.exp(temperature2 * sideways_penalty) - 1.0

    # Total reward with normalized ranges for effective learning
    total_reward = (forward_reward_weight * transformed_forward_reward +
                    sideways_penalty_weight * transformed_sideways_penalty +
                    0.1 * heading_reward)

    # Return the total reward and a dictionary of reward components
    return total_reward, {
        "forward_reward": forward_reward,
        "sideways_penalty": sideways_penalty,
        "heading_reward": heading_reward,
        "transformed_forward_reward": transformed_forward_reward,
        "transformed_sideways_penalty": transformed_sideways_penalty
    }
