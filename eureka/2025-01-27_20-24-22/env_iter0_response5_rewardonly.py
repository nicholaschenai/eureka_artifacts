@torch.jit.script
def compute_reward(hand_pos: torch.Tensor, hand_rot: torch.Tensor, drawer_pos: torch.Tensor, 
                   drawer_rot: torch.Tensor, drawer_open_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for the exponential transformations
    distance_temp = 0.1
    alignment_temp = 0.1
    opening_temp = 0.1

    # Compute distance between hand and drawer handle
    distance_to_drawer = torch.norm(hand_pos - drawer_pos, dim=-1)
    distance_reward = torch.exp(-distance_temp * distance_to_drawer)

    # Compute alignment reward between hand orientation and drawer handle orientation
    # Assuming we use dot product for simplicity as a measure of alignment
    alignment = torch.sum(hand_rot * drawer_rot, dim=-1)  # Similarity as cosine similarity
    alignment_reward = torch.exp(alignment_temp * alignment)

    # Reward for the degree of drawer opening
    drawer_open_degree = drawer_open_pos[:, 0]  # Assuming the X axis is the opening axis
    opening_reward = torch.exp(opening_temp * drawer_open_degree)

    # Total reward
    total_reward = distance_reward + alignment_reward + opening_reward
    total_reward = total_reward.clamp(max=1.0)  # Normalizing the reward to [0,1]

    # Create info dictionary
    info = {
        "distance_reward": distance_reward,
        "alignment_reward": alignment_reward,
        "opening_reward": opening_reward
    }

    return total_reward, info
