import numpy as np
from emotions import N_EMOTION, Emotion
from utils import softmax
class EmotionState():
    """
    Implement the belief state on emotion

    """

    def __init__(self, belief_proba=None):
        if belief_proba is None:
            self.belief_proba = np.ones(N_EMOTION)/N_EMOTION
        else:
            self.belief_proba = belief_proba

        self.emotions = [Emotion(i) for i in range(N_EMOTION)]

    def distance_to(self, other_state):
        distance = np.sum(np.abs(self.belief_proba - other_state.belief_proba))

        return distance

    def copy(self):
        return EmotionState(self.belief_proba.copy() )

    def equals(self, other_state):
        if self.belief_proba == other_state.belief_proba:
            return 1
        else:
            return 0

    def to_string(self):
        state_string = "The most probable emotion is "
        state_string += self.emotions[np.argmax(self.belief_proba)].to_string()
        return state_string

    def print_state(self):
        """
        Pretty printing
        :return:
        """
        print(self.to_string())

    def next_belief_state(self,observation, action, current_state,next_state,transition):


        O = softmax([-e.distance_to_observation(observation) for e in self.emotions])

        B = []
        for i,o in enumerate(O):
            temp = 0
            for e in self.emotions:
                temp += transition[e.bin_number][current_state.as_tuple()][action.bin_number,i][next_state.as_tuple()]*self.belief_proba[e.bin_number]
            B.append(temp*o)
        B = B/np.sum(B)
        return EmotionState(belief_proba=B)
