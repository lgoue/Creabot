from action import Action, N_ACTION
import numpy as np
from emotions import N_EMOTION,Emotion
from EmotionState import EmotionState
from utils import softmax
from Script import Script, N_IDEA_QUALITY, N_AGENT_DA

class Agent():

    def __init__(self,n_time=4,gamma=0.9,wp=0.8,eps_alpha=0.02):

        #self.mean_samples_emotions = np.zeros(N_EMOTION)
        #self.mean_emotion_transitions = np.ones((N_EMOTION,N_ACTION,N_EMOTION))/N_EMOTION
        self.mean_transitions = np.ones((N_EMOTION,N_ACTION,n_time+1,N_AGENT_DA,N_IDEA_QUALITY,N_ACTION,N_EMOTION,N_ACTION,n_time+1,N_AGENT_DA,N_IDEA_QUALITY))/(N_EMOTION*N_ACTION*(n_time+1)*N_AGENT_DA*N_IDEA_QUALITY)
        self.mean_samples = np.zeros((N_EMOTION,N_ACTION,n_time+1,N_AGENT_DA,N_IDEA_QUALITY))

        self.ia = 0
        self.n_time = n_time
        """
        for s in range(N_EMOTION):
            for a in range(N_ACTION):
                for sp in range(N_EMOTION):
                    self.mean_emotion_transitions[s,a,sp] = - Emotion(sp).distance_to(Action(a).occ_emotion(s))
                self.mean_emotion_transitions[s,a] = softmax(self.mean_emotion_transitions[s,a])

        """
        #self.current_user_sample_emotions = np.zeros(N_EMOTION)
        #self.current_user_emotion_transitions = np.ones((N_EMOTION,N_ACTION,N_EMOTION))/N_EMOTION

        self.current_user_transitions = np.ones((N_EMOTION,N_ACTION,n_time+1,N_AGENT_DA,N_IDEA_QUALITY,N_ACTION,N_EMOTION,N_ACTION,n_time+1,N_AGENT_DA,N_IDEA_QUALITY))/(N_EMOTION*N_ACTION*(n_time+1)*N_AGENT_DA*N_IDEA_QUALITY)
        self.current_user_sample = np.zeros((N_EMOTION,N_ACTION,n_time+1,N_AGENT_DA,N_IDEA_QUALITY))


        self.alpha = np.zeros((N_EMOTION,N_ACTION,self.n_time+1,N_AGENT_DA,N_IDEA_QUALITY,N_ACTION))
        self.previous_alpha = np.zeros((N_EMOTION,N_ACTION))

        self.wp=wp
        self.gamma=gamma
        self.script = Script()
        self.eps_alpha = eps_alpha

        self.actions = [Action(a) for a in range(N_ACTION)]

        self.emotions =[Emotion(e) for e in range(N_EMOTION)]
        self.emotion_belief = EmotionState()



    def new_user(self,start_state,get_reward,next_state):
        self.script = Script()
        self.current_user_sample_emotions = np.zeros(N_EMOTION)
        self.emotion_belief = EmotionState()
        self.current_user_transitions = self.mean_transitions.copy()
        #self.transitions = self.wp * self.current_user_transitions + (1-self.wp)*self.mean_transitions

    def update(self,observation, action, state, get_reward,next_state):
        emotion = self.emotion_belief.next_belief_state(observation, action, state,next_state, self.mean_transitions )#self.transitions)

        self.update_current_user_transition(state,next_state,action,emotion.belief_proba)
        self.update_transition()
        self.update_alpha(state,action,get_reward,next_state)
        self.emotion_belief = emotion


    '''
    def update_current_user_emotion_transition(self,state,action,emotion,belief=True):

        if belief:
            for previous_e,previous_b in enumerate(self.emotion_belief.belief_proba):
                for e,b in enumerate(emotion):
                    self.current_user_emotion_transitions[previous_e,action.bin_number,e] = (self.current_user_sample_emotions[previous_e]*self.current_user_emotion_transitions[previous_e,action.bin_number,e] + b*previous_b)/(self.current_user_sample_emotions[previous_e]+previous_b)
                self.current_user_sample_emotions[previous_e] += previous_b
        else :
            previous_e= np.argmax(self.emotion_belief.belief_proba)
            e =np.argmax(emotion)
            self.current_user_emotion_transitions[previous_e,action.bin_number,e] = (self.current_user_sample_emotions[previous_e]*self.current_user_emotion_transitions[previous_e,action.bin_number,e] + emotion[e])/(self.current_user_sample_emotions[previous_e]+1)
            self.current_user_sample_emotions[previous_e] += 1
    '''
    def update_current_user_transition(self,state,next_state,action,emotion,belief=False):
        state_tuple = state.as_tuple()
        next_state_tuple = next_state.as_tuple()
        if belief:
            for previous_e,previous_b in enumerate(self.emotion_belief.belief_proba):
                for e in range(N_EMOTION):

                    self.current_user_transitions[previous_e][state_tuple][action.bin_number,e][next_state_tuple] = (self.current_user_sample[previous_e][state_tuple]*self.current_user_transitions[previous_e][state_tuple][action.bin_number,e][next_state_tuple]  + emotion[e]*previous_b)/(self.current_user_sample[previous_e][state_tuple]+previous_b)
                self.current_user_sample[previous_e][state_tuple] += previous_b
        else :
            previous_e= np.argmax(self.emotion_belief.belief_proba)
            e =np.argmax(emotion)
            self.current_user_transitions[previous_e][state_tuple][action.bin_number,e][next_state_tuple]  = (self.current_user_sample[previous_e][state_tuple]*self.current_user_transitions[previous_e][state_tuple][action.bin_number,e][next_state_tuple]  + emotion[e])/(self.current_user_sample[previous_e][state_tuple]+1)
            self.current_user_sample[previous_e][state_tuple] += 1

    '''
    def update_mean_emotion_transition(self,state,action,emotion):
        self.mean_emotion_transitions = (self.ia*self.mean_emotion_transitions.copy() + self.current_user_emotion_transitions.copy())/(self.ia+1)

    def update_emotion_transition(self):

        self.emotion_transitions = self.wp * self.current_user_emotion_transitions.copy() + (1-self.wp)*self.mean_emotion_transitions.copy()
    '''
    def update_mean_transition(self,state,action,emotion):
        self.mean_transitions = (self.ia*self.mean_transitions.copy() + self.current_user_transitions.copy())/(self.ia+1)

    def update_transition(self):
        pass
        #self.transitions = self.wp * self.current_user_transitions.copy() + (1-self.wp)*self.mean_transitions.copy()

    def get_action(self,state):
        self.Q = np.zeros(N_ACTION)
        for a in self.actions:
            for e in range(N_EMOTION):
                self.Q[a.bin_number] += self.emotion_belief.belief_proba[e]*self.alpha[e][state.as_tuple()][a.bin_number]
        return self.actions[np.argmax(self.Q)]

    def update_alpha(self,state,action,get_reward,next_state):
        self.previous_alpha = self.alpha
        for emotion in self.emotions:
            s = state.copy()
            state.emotion = emotion
            a = get_reward(s,action)

            for e in range(N_EMOTION):
                a += self.gamma*self.mean_transitions[emotion.bin_number][state.as_tuple()][action.bin_number,e][next_state.as_tuple()]*np.max(self.alpha[emotion.bin_number][state.as_tuple()])

                #a += self.gamma*self.transitions[emotion.bin_number][state.as_tuple()][action.bin_number,next_state.emotion.bin_number][next_state.as_tuple()]*np.max(self.alpha[emotion.bin_number][state.as_tuple()])

            self.alpha[emotion.bin_number][state.as_tuple()][action.bin_number] =a
        while np.linalg.norm(self.alpha - self.previous_alpha)>self.eps_alpha:
            for e in self.emotions:
                self.alpha[emotion.bin_number][state.as_tuple()][action.bin_number] = self.get_reward(emotion,action) + self.gamma*np.sum([self.transitions[emotion.bin_number][state.as_tuple][action.bin_number,e][next_state.as_tuple]*np.max(self.alpha[e][state.as_tuple()]) for e in range(N_EMOTION)])
