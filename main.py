import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt

class QLAgent:
    def __init__(self, n_actions=2, learning_rate=0.1, discount_factor=1.00, epsilon=0.1, epsilon_min=0.01, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # obs space is the player sum, dealer's card, and usable ace
        # player sum ranges from 4 to 21 according to docs
        self.q_table = np.zeros([18] + [10] + [2] + [self.n_actions])

    def get_action(self, state, training=True):
        # Epsilon-greedy action selection
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state[0]-4, state[1]-1, int(state[2])])

    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state[0]-4, state[1]-1, int(state[2]), action]

        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state[0]-4, next_state[1]-1, int(next_state[2])])
            target_q = reward + self.gamma * max_next_q

        self.q_table[state[0]-4, state[1]-1, int(state[2]), action] += self.learning_rate * (target_q - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(episodes = 1000, render = False):
    env = gym.make("Blackjack-v1", render_mode="human" if render else None)
    action_size = int(env.action_space.n)  
    agent = QLAgent(n_actions=action_size)
    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # uses epsilon-greedy action selection
            action = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update the Q-table based on the action taken and the reward received
            agent.update(state,action,reward,next_state,done)

            state = next_state
            total_reward += reward

        # Decay epsilon after episode
        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if(episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{episodes}, Average Reward: {np.mean(episode_rewards[-1000:])}")

    env.close()
    print(f"Training complete\n")
    return agent, episode_rewards



def test(agent, episodes = 100, render = True, video_folder = "videos"):
    env = gym.make("Blackjack-v1", render_mode="rgb_array" if render else None)
    if render:
        # Records every episode as its own mp4 in video_folder
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda ep: True, name_prefix="blackjack-test")
    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)

    env.close()
    print(f"Testing complete\n")
    return episode_rewards

# Values
# Baseline:
learning_rate = 0.1
gamma = 1.0
epsilon = 0.1

# best
learning_rate = 0.9
gamma = 1.0
epsilon = 0.01

agent = QLAgent(learning_rate=learning_rate, discount_factor=gamma, epsilon=epsilon)
agent, episode_rewards = train(episodes=100000, render=False)

def moving_average(x, window):
    return np.convolve(x, np.ones(window) / window, mode="valid")

window = 5000
ma = moving_average(episode_rewards, window)
plt.plot(np.arange(window, len(episode_rewards) + 1), ma)

plt.xlabel("Episode")
plt.ylabel(f"Average reward (last {window} episodes)")
plt.title("Training Progress")
plt.show()


episode_rewards_test = test(agent, episodes=30, render=True)

successes = sum(1 for r in episode_rewards_test if r > 0)
print(f"Number of successful episodes: {successes}/{len(episode_rewards_test)}, {successes/len(episode_rewards_test):.2f}")


window = 3
ma = moving_average(episode_rewards_test, window)
plt.plot(np.arange(window, len(episode_rewards_test) + 1), ma)

plt.xlabel("Episode")
plt.ylabel(f"Average reward (last {window} episodes)")
plt.title("Testing Progress")
plt.show()

import glob, re
try:
    from moviepy import VideoFileClip, concatenate_videoclips  # moviepy 2.x
except ImportError:
    from moviepy.editor import VideoFileClip, concatenate_videoclips  # moviepy 1.x

files = sorted(glob.glob("videos/blackjack-test-episode-*.mp4"),
               key=lambda f: int(re.search(r"episode-(\d+)", f).group(1)))
clips = [VideoFileClip(f) for f in files]
concatenate_videoclips(clips).write_videofile("videos/blackjack-all.mp4")
for c in clips:
    c.close()

# Learning rate variations: 0.01, 0.5, 0.9
#agent1 = QLAgent(learning_rate=0.01, discount_factor=gamma, epsilon=epsilon)
#agent1, episode_rewards1 = train(episodes=100000, render=False)

#agent2 = QLAgent(learning_rate=0.5, discount_factor=gamma, epsilon=epsilon)
#agent2, episode_rewards2 = train(episodes=100000, render=False)

#agent3 = QLAgent(learning_rate=0.9, discount_factor=gamma, epsilon=epsilon)
#agent3, episode_rewards3 = train(episodes=100000, render=False)

# gamma variations: 0.5, 0.9, 1.0
#agent1 = QLAgent(learning_rate=learning_rate, discount_factor=0.5, epsilon=epsilon)
#agent1, episode_rewards1 = train(episodes=100000, render=False)

#agent2 = QLAgent(learning_rate=learning_rate, discount_factor=0.9, epsilon=epsilon)
#agent2, episode_rewards2 = train(episodes=100000, render=False)

#agent3 = QLAgent(learning_rate=learning_rate, discount_factor=1.0, epsilon=epsilon)
#agent3, episode_rewards3 = train(episodes=100000, render=False)


# epsilon variations: 0.01, 0.5, 0.9
#agent1 = QLAgent(learning_rate=learning_rate, discount_factor=gamma, epsilon=0.01)
#agent1, episode_rewards1 = train(episodes=100000, render=False)

#agent2 = QLAgent(learning_rate=learning_rate, discount_factor=gamma, epsilon=0.5)
#agent2, episode_rewards2 = train(episodes=100000, render=False)

#agent3 = QLAgent(learning_rate=learning_rate, discount_factor=gamma, epsilon=0.9)
#agent3, episode_rewards3 = train(episodes=100000, render=False)


# Plot the episode rewards
#def moving_average(x, window):
#    return np.convolve(x, np.ones(window) / window, mode="valid")

#window = 5000
#ma1 = moving_average(episode_rewards1, window)
#plt.plot(np.arange(window, len(episode_rewards1) + 1), episode_rewards1[window-1:])
#ma2 = moving_average(episode_rewards2, window)
#plt.plot(np.arange(window, len(episode_rewards2) + 1), episode_rewards2[window-1:])
#ma3 = moving_average(episode_rewards3, window)
#plt.plot(np.arange(window, len(episode_rewards3) + 1), episode_rewards3[window-1:])
#plt.legend(["Agent 1 (epsilon=0.01)", "Agent 2 (epsilon=0.5)", "Agent 3 (epsilon=0.9)"])

#plt.xlabel("Episode")
#plt.ylabel(f"Average reward (last {window} episodes)")
#plt.title("Training Progress")
#plt.show()