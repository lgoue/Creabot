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
from moods import N_MOOD, Mood, MoodType
from agent import Agent
from Script import AgentDa, UserDa, DialogActAgent, Idea, IdeaQuality, N_AGENT_DA,N_IDEA_QUALITY

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
        self.agent.new_user(self.get_start_state(),1//self.inte_t)

    def get_start_state(self):
        return CreabotState(Mood(MoodType.NEUTRAL),ActionType.NEUTRAL,0,AgentDa(DialogActAgent.GREETING),Idea(IdeaQuality.NO))
    def all_states(self):
        states = []
        for m in range(N_MOOD):
            #for next_da in self.da.possible_next_da():
            for score in self.da.possible_idea_quality():
                for time in range(self.NTime+1):
                    for a in range(N_ACTION):
                        states.append(CreabotState(Mood(m),a,time,AgentDa(0),Idea(score)))
        return states
    def interaction_turn(self,state, verbose = True):
        print("agent :",AgentDa(self.agent.script.current_state).to_string())
        action = self.agent.get_action(state)

        self.perform_action(state, action)
        print("user", UserDa(self.user.da).to_string())
        observation = self.get_observation()


        is_terminal,next_state = self.get_next_state(state,action)


        reward_no_entro = self.get_reward_no_entro(state,action, next_state)
        reward = self.get_reward(state,action, next_state)

        self.agent.update(observation,action, state,reward, next_state)


        if(verbose):

            #observation.print_observation()
            if state.emotion is not None:
                self.agent.emotion_belief.print_state()
            else:
                self.agent.mood_belief.print_state()
            print(action.to_string())
            self.user.print_emotion()
            self.user.print_mood()
            next_state.print()
        return is_terminal, next_state, action, reward_no_entro



    def interaction(self,verbose = True):

        self.reset_episode()
        state = self.get_start_state()
        is_terminal = False
        emotion_error = []
        emotion_acc = []
        mood_error = []
        mood_acc = []
        belief_error = []
        actions = []
        cumul_reward = 0

        while not is_terminal :

            is_terminal, state,action, reward = self.interaction_turn(state,verbose=verbose)
            belief_error.append(np.linalg.norm(self.agent.emotion_belief.belief_proba- self.user.Es))
            if state.emotion is not None:
                emotion_error.append(Emotion(np.argmax(self.agent.emotion_belief.belief_proba)).distance_to(Emotion(np.argmax(self.user.Es))))
                emotion_acc.append(np.argmax(self.agent.emotion_belief.belief_proba) == np.argmax(self.user.Es))
            else:
                #mood_error.append(Mood(np.argmax(self.agent.mood_belief.belief_proba)).distance_to(Mood(np.argmax(self.user.Es))))
                mood_acc.append(np.argmax(self.agent.mood_belief.belief_proba) == np.argmax(self.user.Es))
            actions.append(action.bin_number)
            cumul_reward += reward

        self.agent.update_mean_transition()
        self.agent.update_mean_reward()
        self.agent.ia +=1
        if state.emotion is not None:
            return emotion_error, emotion_acc, belief_error,actions, cumul_reward
        else:
            return mood_error, mood_acc, belief_error,actions, cumul_reward


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

    def get_next_state(self,state,action):

        time = self.time
        is_terminal = time == self.Tmax
        if state.emotion is not None:
            emotion_state = Emotion(np.argmax(self.agent.emotion_belief.belief_proba))
        else:
            mood_state = Mood(np.argmax(self.agent.mood_belief.belief_proba))
        self.agent.script.update_state(self.user.da)

        idea_quality = self.user.get_idea_quality()
        if state.emotion is not None:
            return is_terminal,CreabotState(emotion_state,action.bin_number,time // self.inte_t,AgentDa(self.agent.script.current_state),idea_quality)
        else:
            return is_terminal,CreabotState(mood_state,action.bin_number,time // self.inte_t,AgentDa(self.agent.script.current_state),idea_quality)


    def get_reward(self,state,action,next_state):

        if state.emotion is not None:
            print("error")
        else:
            H = - np.sum(entropy(self.agent.transitions[state.mood.bin_number][state.as_tuple()][action.bin_number]))
            reward = next_state.idea_score.quality*10 + H/100 -1*(action.bin_number == state.last_strat)
        return reward
    def get_reward_no_entro(self,state,action,next_state):

        if state.emotion is not None:
            print("error")
        else:
            H = - np.sum(entropy(self.agent.transitions[state.mood.bin_number][state.as_tuple()][action.bin_number]))
            reward = next_state.idea_score.quality*10 -1*(action.bin_number == state.last_strat)
        return reward
