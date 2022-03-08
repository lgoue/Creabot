from action import Action, N_ACTION
import numpy as np
from emotions import N_EMOTION,Emotion
from EmotionState import EmotionState
from MoodState import MoodState
from utils import softmax
from moods import N_MOOD,Mood
from Script import Script, N_IDEA_QUALITY, N_AGENT_DA

class Agent():

    def __init__(self,n_time=4,gamma=0.9,wp=0.8,eps_alpha=0.002):

        #self.mean_samples_emotions = np.zeros(N_EMOTION)
        #self.mean_emotion_transitions = np.ones((N_EMOTION,N_ACTION,N_EMOTION))/N_EMOTION
        self.mean_transitions = np.ones((N_MOOD,N_ACTION,n_time+1,N_AGENT_DA,N_IDEA_QUALITY,N_ACTION,N_MOOD,N_ACTION,n_time+1,N_AGENT_DA,N_IDEA_QUALITY))/(N_MOOD*N_ACTION*(n_time+1)*N_AGENT_DA*N_IDEA_QUALITY)
        self.mean_samples = np.zeros((N_MOOD,N_ACTION,n_time+1,N_AGENT_DA,N_IDEA_QUALITY))

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

        self.current_user_transitions = np.ones((N_MOOD,N_ACTION,n_time+1,N_AGENT_DA,N_IDEA_QUALITY,N_ACTION,N_MOOD,N_ACTION,n_time+1,N_AGENT_DA,N_IDEA_QUALITY))/(N_MOOD*N_ACTION*(n_time+1)*N_AGENT_DA*N_IDEA_QUALITY)
        self.current_user_sample = np.zeros((N_MOOD,N_ACTION,n_time+1,N_AGENT_DA,N_IDEA_QUALITY))


        self.alpha = np.ones((N_MOOD,N_ACTION,self.n_time+1,N_AGENT_DA,N_IDEA_QUALITY,N_ACTION))/(N_MOOD*N_ACTION*(n_time+1)*N_AGENT_DA*N_IDEA_QUALITY)
        self.previous_alpha = np.ones((N_MOOD,N_ACTION,self.n_time+1,N_AGENT_DA,N_IDEA_QUALITY,N_ACTION))/(N_MOOD*N_ACTION*(n_time+1)*N_AGENT_DA*N_IDEA_QUALITY)

        self.wp=wp
        self.gamma=gamma
        self.script = Script()
        self.eps_alpha = eps_alpha

        self.actions = [Action(a) for a in range(N_ACTION)]

        self.emotions =[Emotion(e) for e in range(N_EMOTION)]
        self.emotion_belief = EmotionState()

        self.moods =[Mood(m) for m in range(N_MOOD)]
        self.mood_belief = MoodState()


    def new_user(self,start_state,get_reward,next_state):
        self.script = Script()
        self.current_user_sample = np.zeros((N_MOOD,N_ACTION,self.n_time+1,N_AGENT_DA,N_IDEA_QUALITY))
        self.emotion_belief = EmotionState()
        self.mood_belief = MoodState()
        self.current_user_transitions = self.mean_transitions.copy()
        self.transitions = self.wp * self.current_user_transitions.copy() + (1-self.wp)*self.mean_transitions.copy()

    def update(self,observation, action, state, get_reward,next_state):
        if state.emotion is not None:
            emotion = self.emotion_belief.next_belief_state(observation, action, state,next_state, self.transitions )
            self.update_current_user_transition(state,next_state,action,None,emotion=emotion.belief_proba)
            self.update_transition()
            self.update_alpha(state,action,get_reward,next_state)
            self.emotion_belief = emotion
        else :
            mood = self.mood_belief.next_belief_state(observation, action, state,next_state, self.transitions )
            self.update_current_user_transition(state,next_state,action,mood.belief_proba)
            self.update_transition()
            self.update_alpha(state,action,get_reward,next_state)
            self.mood_belief = mood
            print("check :", np.sum(self.transitions)/(N_MOOD*N_ACTION*(self.n_time+1)*N_AGENT_DA*N_IDEA_QUALITY*N_ACTION))
            print("check :", np.sum(self.current_user_transitions)/(N_MOOD*N_ACTION*(self.n_time+1)*N_AGENT_DA*N_IDEA_QUALITY*N_ACTION))




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
    def update_current_user_transition(self,state,next_state,action,mood,emotion=None,belief=False):
        if state.emotion is not None:
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
        else:
            state_tuple = state.as_tuple()
            next_state_tuple = next_state.as_tuple()
            if belief:
                for previous_m,previous_b in enumerate(self.mood_belief.belief_proba):
                    for m in range(N_MOOD):

                        self.current_user_transitions[previous_m][state_tuple][action.bin_number,m][next_state_tuple] = (self.current_user_sample[previous_m][state_tuple]*self.current_user_transitions[previous_m][state_tuple][action.bin_number,e][next_state_tuple]  + mood[m]*previous_b)/(self.current_user_sample[previous_m][state_tuple]+previous_b)
                    self.current_user_sample[previous_m][state_tuple] += previous_b
            else :
                previous_m= np.argmax(self.mood_belief.belief_proba)
                m =np.argmax(mood)
                self.current_user_transitions[previous_m][state_tuple][action.bin_number,m][next_state_tuple]  = (self.current_user_sample[previous_m][state_tuple]*self.current_user_transitions[previous_m][state_tuple][action.bin_number,m][next_state_tuple]  + mood[m])/(self.current_user_sample[previous_m][state_tuple]+1)
                self.current_user_sample[previous_m][state_tuple] += 1

    '''
    def update_mean_emotion_transition(self,state,action,emotion):
        self.mean_emotion_transitions = (self.ia*self.mean_emotion_transitions.copy() + self.current_user_emotion_transitions.copy())/(self.ia+1)

    def update_emotion_transition(self):

        self.emotion_transitions = self.wp * self.current_user_emotion_transitions.copy() + (1-self.wp)*self.mean_emotion_transitions.copy()
    '''
    def update_mean_transition(self):
        print("update mean trans")
        self.mean_transitions = (self.ia*self.mean_transitions.copy() + self.current_user_transitions.copy())/(self.ia+1)

    def update_transition(self):

        self.transitions = self.wp * self.current_user_transitions.copy() + (1-self.wp)*self.mean_transitions.copy()

    def get_action(self,state,beta_explor = 0.8):
        print("max trans",np.max(self.transitions))
        if state.emotion is not None:
            self.Q = np.zeros(N_ACTION)
            for a in self.actions:
                for e in range(N_EMOTION):
                    self.Q[a.bin_number] += self.emotion_belief.belief_proba[e]*self.alpha[e][state.as_tuple()][a.bin_number]
            return self.actions[np.argmax(self.Q)]
        else:

            self.Q = np.zeros(N_ACTION)
            for a in self.actions:
                for m in range(N_MOOD):
                    self.Q[a.bin_number] += self.mood_belief.belief_proba[m]*self.alpha[m][state.as_tuple()][a.bin_number]
            p = softmax(self.Q*beta_explor)
            r = np.random.rand()
            print(np.sum(p))
            n = len(p)
            tot_p = p[0]
            i=0
            while r > tot_p and i<n-1:
                i+=1
                tot_p+=p[i]
            return self.actions[i]

    def update_alpha(self,state,action,get_reward,next_state):

        if state.emotion is not None:
            diff_alpha = 100
            while diff_alpha>self.eps_alpha:

                self.previous_alpha = self.alpha.copy()
                for emotion in self.emotions:
                    a = 0
                    s = state.copy()
                    s.emotion = emotion
                    for e in self.emotions:
                        sp = next_state.copy()
                        sp.emotion = e
                        a += self.transitions[s.emotion.bin_number][state.as_tuple()][action.bin_number,sp.emotion.bin_number][sp.as_tuple()]*(get_reward(s,action,sp) + self.gamma*np.max(self.alpha[sp.emotion.bin_number][sp.as_tuple()]))


                    self.alpha[s.emotion.bin_number][state.as_tuple()][action.bin_number] =a
                diff_alpha=np.linalg.norm(self.alpha - self.previous_alpha)

        else:
            diff_alpha = 100
            while diff_alpha>self.eps_alpha:

                self.previous_alpha = self.alpha.copy()
                for mood in self.moods:
                    a = 0
                    s = state.copy()
                    s.mood = mood
                    for m in self.moods:

                        sp = next_state.copy()
                        sp.mood = m
                        a += self.transitions[s.mood.bin_number][state.as_tuple()][action.bin_number,sp.mood.bin_number][sp.as_tuple()]*(get_reward(s,action,sp) + self.gamma*np.max(self.alpha[sp.mood.bin_number][sp.as_tuple()]))


                    self.alpha[s.mood.bin_number][state.as_tuple()][action.bin_number] =a
                diff_alpha=np.linalg.norm(self.alpha - self.previous_alpha)
