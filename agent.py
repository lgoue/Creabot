import numpy as np
from action import N_ACTION, Action
from Creabotstate import CreabotState
from emotions import N_EMOTION, Emotion
from EmotionState import EmotionState
from moods import N_MOOD, Mood
from MoodState import MoodState
from Script import N_AGENT_DA, AgentDa, Idea, Script
from utils import softmax


class Agent:
    def __init__(self, n_time=4, gamma=0.9, wp=0.8, eps_alpha=0.002, beta=0.8):

        self.mean_transitions = np.ones(
            (N_MOOD, N_ACTION, n_time + 1, N_ACTION, N_MOOD, N_ACTION, n_time + 1)
        ) / (N_MOOD * N_ACTION * (n_time + 1))

        self.mean_reward = np.ones(
            (N_MOOD, N_ACTION, n_time + 1, N_ACTION, N_MOOD, N_ACTION, n_time + 1)
        ) / (N_MOOD * N_ACTION * (n_time + 1))

        self.ia = 0
        self.n_time = n_time

        self.beta = beta

        states = []
        for m in range(N_MOOD):

            for time in range(self.n_time + 1):
                for a in range(N_ACTION):
                    states.append(CreabotState(Mood(m), a, time, AgentDa(0), Idea(0)))
        self.all_states = states

        self.current_user_transitions = np.ones(
            (N_MOOD, N_ACTION, n_time + 1, N_ACTION, N_MOOD, N_ACTION, n_time + 1)
        ) / (N_MOOD * N_ACTION * (n_time + 1))
        self.current_user_sample = np.ones((N_MOOD, N_ACTION, n_time + 1, N_ACTION))
        self.current_user_reward = np.ones(
            (N_MOOD, N_ACTION, n_time + 1, N_ACTION, N_MOOD, N_ACTION, n_time + 1)
        ) / (N_MOOD * N_ACTION * (n_time + 1))
        self.current_user_reward_sample = np.ones(
            (
                N_MOOD,
                N_ACTION,
                self.n_time + 1,
                N_ACTION,
                N_MOOD,
                N_ACTION,
                self.n_time + 1,
            )
        )
        self.alpha = np.ones((N_MOOD, N_ACTION, self.n_time + 1, N_ACTION)) / (
            N_MOOD * N_ACTION * (n_time + 1)
        )
        self.previous_alpha = np.ones((N_MOOD, N_ACTION, self.n_time + 1, N_ACTION)) / (
            N_MOOD * N_ACTION * (n_time + 1)
        )

        self.wp = wp
        self.gamma = gamma
        self.script = Script()
        self.eps_alpha = eps_alpha

        self.actions = [Action(a) for a in range(N_ACTION)]

        self.emotions = [Emotion(e) for e in range(N_EMOTION)]
        self.emotion_belief = EmotionState()

        self.moods = [Mood(m) for m in range(N_MOOD)]
        self.mood_belief = MoodState()

    def new_user(self, start_state, next_state):
        self.script = Script()
        self.current_user_sample = np.ones(
            (N_MOOD, N_ACTION, self.n_time + 1, N_ACTION)
        )
        self.current_user_transitions = self.mean_transitions.copy()
        self.current_user_reward = self.mean_reward.copy()
        self.current_user_reward_sample = np.ones(
            (
                N_MOOD,
                N_ACTION,
                self.n_time + 1,
                N_ACTION,
                N_MOOD,
                N_ACTION,
                self.n_time + 1,
            )
        )
        self.emotion_belief = EmotionState()
        self.mood_belief = MoodState()
        self.current_user_transitions = self.mean_transitions.copy()
        self.transitions = (
            self.wp * self.current_user_transitions.copy()
            + (1 - self.wp) * self.mean_transitions.copy()
        )

    def update(self, observation, action, state, reward, next_state):
        if state.emotion is not None:
            print("erorr")
        else:
            mood = self.mood_belief.next_belief_state(
                observation, action, state, next_state, self.transitions
            )
            self.update_current_user_transition(
                state, next_state, action, mood.belief_proba
            )
            self.update_current_user_reward(
                state, next_state, action, mood.belief_proba, reward
            )
            self.update_transition()
            self.update_reward()
            self.update_alpha(state, action, next_state)
            self.mood_belief = mood

    def update_current_user_transition(
        self, state, next_state, action, mood, emotion=None, belief=True
    ):
        if state.emotion is not None:
            print("errrorrrr")
        else:
            state_tuple = state.as_tuple()
            next_state_tuple = next_state.as_tuple()
            if belief:

                for previous_m, previous_b in enumerate(self.mood_belief.belief_proba):
                    self.current_user_transitions[previous_m][state_tuple][
                        action.bin_number
                    ] *= self.current_user_sample[previous_m][state_tuple][
                        action.bin_number
                    ]
                    for m in range(N_MOOD):
                        self.current_user_transitions[previous_m][state_tuple][
                            action.bin_number, m
                        ][next_state_tuple] = (
                            self.current_user_transitions[previous_m][state_tuple][
                                action.bin_number, m
                            ][next_state_tuple]
                            + mood[m] * previous_b
                        )
                    self.current_user_transitions[previous_m][state_tuple][
                        action.bin_number
                    ] /= (
                        self.current_user_sample[previous_m][state_tuple][
                            action.bin_number
                        ]
                        + previous_b
                        + 0.00001
                    )
                    self.current_user_sample[previous_m][state_tuple][
                        action.bin_number
                    ] += previous_b

            else:
                print("errrorrr")

    def update_current_user_reward(
        self, state, next_state, action, mood, reward, emotion=None, belief=True
    ):
        if state.emotion is not None:
            print("errrorrrr")
        else:
            state_tuple = state.as_tuple()
            next_state_tuple = next_state.as_tuple()
            if belief:
                for previous_m, previous_b in enumerate(self.mood_belief.belief_proba):
                    for m in range(N_MOOD):
                        self.current_user_reward[previous_m][state_tuple][
                            action.bin_number, m
                        ][next_state_tuple] += (reward * previous_b * mood[m])
                    self.current_user_reward[previous_m][state_tuple][
                        action.bin_number, m
                    ][next_state_tuple] /= (
                        self.current_user_reward_sample[previous_m][state_tuple][
                            action.bin_number, m
                        ][next_state_tuple]
                        + previous_b * mood[m]
                        + 0.00001
                    )
                    self.current_user_reward_sample[previous_m][state_tuple][
                        action.bin_number, m
                    ][next_state_tuple] += (previous_b * mood[m])

            else:
                print("errrorrr")

    def update_mean_transition(self):

        self.mean_transitions = (
            self.ia * self.mean_transitions.copy()
            + self.current_user_transitions.copy()
        ) / (self.ia + 1)

    def update_transition(self):

        self.transitions = (
            self.wp * self.current_user_transitions.copy()
            + (1 - self.wp) * self.mean_transitions.copy()
        )

    def update_mean_reward(self):

        self.mean_reward = (
            self.ia * self.mean_reward.copy() + self.current_user_reward.copy()
        ) / (self.ia + 1)

    def update_reward(self):

        self.reward = (
            self.wp * self.current_user_reward.copy()
            + (1 - self.wp) * self.mean_reward.copy()
        )

    def get_action(self, state):

        if state.emotion is not None:
            self.Q = np.zeros(N_ACTION)
            for a in self.actions:
                for e in range(N_EMOTION):
                    self.Q[a.bin_number] += (
                        self.emotion_belief.belief_proba[e]
                        * self.alpha[e][state.as_tuple()][a.bin_number]
                    )
            return self.actions[np.argmax(self.Q)]
        else:

            self.Q = np.zeros(N_ACTION)
            for a in self.actions:
                for m in range(N_MOOD):
                    self.Q[a.bin_number] += (
                        self.mood_belief.belief_proba[m]
                        * self.alpha[m][state.as_tuple()][a.bin_number]
                    )
            p = softmax(self.Q * self.beta)
            r = np.random.rand()

            n = len(p)
            tot_p = p[0]
            i = 0
            while r > tot_p and i < n - 1:
                i += 1
                tot_p += p[i]
            return self.actions[i]

    def update_alpha(self, state, action, next_state):

        if state.emotion is not None:
            print("errrorrr")

        else:
            diff_alpha = 100
            while diff_alpha > self.eps_alpha:

                self.previous_alpha = self.alpha.copy()
                for mood in self.moods:
                    a = 0
                    s = state.copy()
                    s.mood = mood

                    for sp in self.all_states:
                        a += self.transitions[s.mood.bin_number][state.as_tuple()][
                            action.bin_number, sp.mood.bin_number
                        ][sp.as_tuple()] * (
                            self.mean_reward[s.mood.bin_number][s.as_tuple()][
                                action.bin_number
                            ][sp.mood.bin_number][sp.as_tuple()]
                            + self.gamma
                            * np.max(self.alpha[sp.mood.bin_number][sp.as_tuple()])
                        )

                    self.alpha[s.mood.bin_number][state.as_tuple()][
                        action.bin_number
                    ] = a
                diff_alpha = np.linalg.norm(self.alpha - self.previous_alpha)
