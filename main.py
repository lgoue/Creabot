from interaction import InteractionModel
import matplotlib.pyplot as plt
from utils import entropy
from action import ActionType
from emotions import EmotionType
import numpy as np
if __name__ == '__main__':

    env = InteractionModel()
    mean_emotion_error = []
    evol_trans = []
    actions = []
    belief_errors = []
    emotion_errors = []
    emotion_accs = []
    entropies = []
    test = []
    cumul_rewards =[]
    for i in range(1000):
        emotion_error, emotion_acc, belief_error,action, cumul_reward = env.interaction(verbose = True)
        #mean_emotion_error += [np.mean(emotion_error)]
        #actions += action
        #belief_errors += [np.mean(belief_error)]
        #emotion_errors += [np.mean(emotion_error)]
        #emotion_accs += [np.mean(emotion_acc)]
        entropies.append(-np.sum(entropy(env.agent.mean_transitions)))
        #test.append(np.mean(np.array(env.agent.emotion_transitions)[:,ActionType.SELF_DISCLOSE,EmotionType.LOVE]))
        cumul_rewards.append(cumul_reward)


        plt.plot(cumul_rewards, label="reward cumul")
        plt.legend()
        plt.savefig("reward.png")
        plt.show()

    #plt.plot(actions,label = "action")
    #plt.plot(emotion_errors,label = "emotion error")
    #plt.plot(belief_errors, label ="belief error")
        plt.plot(entropies, label="entropy")
    #plt.plot(test, label ="test")
    #plt.plot(emotion_accs, label = "emotion accuracy")
        plt.legend()
        plt.savefig("entropy.png")
        plt.show()
