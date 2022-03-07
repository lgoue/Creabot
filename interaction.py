from action import Action, ActionType
from observation import Observation
from EmotionState import EmotionState
from Creabotstate import CreabotState
import json
import numpy as np
from user import User
from utils import softmax, entropy
import matplotlib.pyplot as plt
from emotions import N_EMOTION, Emotion, EmotionType
from action import N_ACTION
from agent import Agent
from Script import AgentDa, UserDa, DialogActAgent, Idea, IdeaQuality

class InteractionModel():
    def __init__(self):
        # logging utility

        self.creabot_config = json.load(open("creabot_config.json", "r"))
        # -------- Model configurations -------- #

        # The maximum Time
        self.Tmax = self.creabot_config['Tmax']
        self.NTime = self.creabot_config['NTime']
        self.inte_t = self.Tmax // self.NTime
        assert self.Tmax == self.NTime * self.inte_t
        self.reward_P = self.creabot_config['reward_P']

        #------------- Simulated interaction ------------------#
        self.user = User()
        self.agent = Agent(n_time = self.NTime,gamma=self.creabot_config['gamma'],wp=self.creabot_config['wp'],eps_alpha=self.creabot_config['eps_alpha'])
        self.time = 0

    def reset_episode(self):
        self.user = User()
        self.time = 0
        self.agent.new_user(self.get_start_state(),self.get_reward,1//self.inte_t)

    def get_start_state(self):
        return CreabotState(Emotion(EmotionType.NEUTRAL),ActionType.NEUTRAL,0,AgentDa(DialogActAgent.GREETING),Idea(IdeaQuality.NO))

    def interaction_turn(self,state, verbose = True):
        print("agent :",AgentDa(self.agent.script.current_state).to_string())
        action = self.agent.get_action(state)

        self.perform_action(state, action)
        print("user", UserDa(self.user.da).to_string())
        observation = self.get_observation()


        is_terminal,next_state = self.get_next_state(action)
        self.agent.update(observation,action, state, self.get_reward, next_state)

        reward = self.get_reward(state,action)


        if(verbose):

            #observation.print_observation()
            self.agent.emotion_belief.print_state()
            print(action.to_string())
            self.user.print_emotion()
            next_state.print()
        return is_terminal, next_state, action, reward



    def interaction(self,verbose = True):

        self.reset_episode()
        state = self.get_start_state()
        is_terminal = False
        emotion_error = []
        emotion_acc = []
        belief_error = []
        actions = []
        cumul_reward = 0

        while not is_terminal :
            is_terminal, state,action, reward = self.interaction_turn(state,verbose=verbose)
            belief_error.append(np.linalg.norm(self.agent.emotion_belief.belief_proba- self.user.Es))
            emotion_error.append(Emotion(np.argmax(self.agent.emotion_belief.belief_proba)).distance_to(Emotion(np.argmax(self.user.Es))))
            emotion_acc.append(np.argmax(self.agent.emotion_belief.belief_proba) == np.argmax(self.user.Es))
            actions.append(action.bin_number)
            cumul_reward += reward

        self.agent.ia +=1
        return emotion_error, emotion_acc, belief_error,actions, cumul_reward


    def perform_action(self, state,action):

        self.user.updateEmotionalActivation(state, action)
        self.user.updateEmotionalState()
        self.user.updateMood()
        self.time += 1
        self.user.update_da(self.agent.script.get_user_possible_da())
    def get_observation(self):
        P = self.user.get_P()
        A = self.user.get_A()
        D = self.user.get_D()
        return Observation(P,A,D)

    def get_next_state(self,action):

        time = self.time
        is_terminal = time == self.Tmax
        emotion_state = Emotion(np.argmax(self.agent.emotion_belief.belief_proba))
        self.agent.script.update_state(self.user.da)

        idea_quality = self.user.get_idea_quality()
        return is_terminal,CreabotState(emotion_state,action.bin_number,time // self.inte_t,AgentDa(self.agent.script.current_state),idea_quality)


    def get_reward(self,state,action):
        #self.transition
        H = - np.sum(np.sum(entropy(self.agent.mean_transitions[state.emotion.bin_number][state.as_tuple()][action.bin_number])))
        reward = state.emotion.PAD_values()[0] * self.reward_P + H
        return reward
