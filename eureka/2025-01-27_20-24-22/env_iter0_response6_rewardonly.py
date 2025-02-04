@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    franka_dof_vel: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters
    distance_temperature: float = 1.0
    velocity_temperature: float = 0.1
    opening_temperature: float = 0.5

    # Distance from franka's grasp position to drawer's grasp position
    distance_to_drawer = torch.norm(drawer_grasp_pos - franka_grasp_pos, p=2, dim=-1)
    distance_reward = torch.exp(-distance_to_drawer / distance_temperature)

    # Encouraging velocity towards the drawer opening direction, use only positive velocities
    opening_velocity_reward = torch.clamp(cabinet_dof_vel[:, 3], min=0.0)
    opening_velocity_reward = torch.exp(opening_velocity_reward / velocity_temperature)

    # Reward for the position/angle of the cabinet door (encourage opening)
    door_opening_reward = torch.clamp(cabinet_dof_pos[:, 3], min=0.0)  # Assuming max open state is positive
    door_opening_reward = torch.exp(door_opening_reward / opening_temperature)

    # Total reward is the sum of all components
    total_reward = distance_reward + opening_velocity_reward + door_opening_reward

    # Collect individual components into a dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "opening_velocity_reward": opening_velocity_reward,
        "door_opening_reward": door_opening_reward
    }

    return total_reward, reward_components
