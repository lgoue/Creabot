import numpy as np
from moods import N_MOOD, Mood
from utils import softmax


class MoodState:
    """
    Implement the belief state on emotion

    """

    def __init__(self, belief_proba=None):
        if belief_proba is None:
            self.belief_proba = np.ones(N_MOOD) / N_MOOD
        else:
            self.belief_proba = belief_proba

        self.moods = [Mood(i) for i in range(N_MOOD)]

    def distance_to(self, other_state):
        distance = np.sum(np.abs(self.belief_proba - other_state.belief_proba))

        return distance

    def copy(self):
        return MoodState(self.belief_proba.copy())

    def equals(self, other_state):
        if self.belief_proba == other_state.belief_proba:
            return 1
        else:
            return 0

    def to_string(self):
        state_string = "The most probable mood is "
        state_string += self.moods[np.argmax(self.belief_proba)].to_string()
        return state_string

    def print_state(self):
        """
        Pretty printing
        :return:
        """
        print(self.to_string())

    def next_belief_state(
        self, observation, action, current_state, next_state, transition
    ):

        O = softmax([-m.distance_to_observation(observation) for m in self.moods])

        B = []
        for i, o in enumerate(O):
            temp = 0
            for m in self.moods:
                temp += (
                    transition[m.bin_number][current_state.as_tuple()][
                        action.bin_number, i
                    ][next_state.as_tuple()]
                    * self.belief_proba[m.bin_number]
                )
            B.append(temp * o)

        B = B / (np.sum(B) + 0.001)
        return MoodState(belief_proba=B)
