import matplotlib.pyplot as plt
import numpy as np
from action import ActionType
from emotions import EmotionType
from interaction import InteractionModel
from utils import entropy

if __name__ == "__main__":

    env = InteractionModel()
    mean_emotion_error = []
    evol_trans = []
    actions = []
    belief_errors = []
    emotion_errors = []
    emotion_accs = []
    entropies = []
    test = []
    cumul_rewards = []
    for i in range(1000):
        (
            emotion_error,
            emotion_acc,
            belief_error,
            action,
            cumul_reward,
        ) = env.interaction(verbose=True)

        entropies.append(-np.sum(entropy(env.agent.mean_transitions)))

        cumul_rewards.append(cumul_reward)

        plt.plot(cumul_rewards, label="reward cumul")
        plt.legend()
        plt.savefig("reward.png")
        plt.show()

        plt.plot(entropies, label="entropy")
        plt.savefig("entropy.png")
        plt.show()
