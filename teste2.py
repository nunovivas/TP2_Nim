# Initialize the AI
from nim import NimAI


ai = NimAI(alpha=0.19)

# Define a state and action
state = (1, 2, 3)
action = (0, 1)

# Update the Q-value for the state-action pair
old_q = 0.5
reward = 1
future_rewards = 0.5
ai.update_q_value(state, action, old_q, reward, future_rewards)

# Retrieve the Q-value
q_value = ai.get_q_value(state, action)

# Check if the Q-value is within the expected range
expected_q_value = old_q + 0.1 * ((reward + future_rewards) - old_q)
print(q_value)
assert (
    abs(q_value - expected_q_value) < 1e-6
), f"Expected Q-value to be {expected_q_value}, got {q_value}"
