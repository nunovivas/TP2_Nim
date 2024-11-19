# Initialize the AI
from nim import NimAI


ai = NimAI(alpha=0.1)

# Define a state and action
state = (1, 2, 3)
action = (0, 1)

# Update the Q-value for the state-action pair
ai.update_q_value(state, action, old_q=0.5, reward=1, future_rewards=0.5)

# Retrieve the Q-value
q_value = ai.get_q_value(state, action)

# Check if the Q-value is within the expected range
expected_q_value = 0.5 + 0.1 * ((1 + 0.5) - 0.5)
print(f"Q-value: {q_value}, Expected Q-value: {expected_q_value}")
assert (
    0.599 <= q_value <= 0.601
), f"Expected Q-value to be in range [0.599, 0.601], got {q_value}"
