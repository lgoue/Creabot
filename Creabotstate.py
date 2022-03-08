
import numpy as np
from action import ActionType, Action
from Script import AgentDa, Idea
from emotions import N_EMOTION, Emotion
from moods import N_MOOD, Mood


class CreabotState():
    """
    Enumerated state for the Creabot POMDP

    Consists of  :
        - a float "time", the time since the begining of the interaction wich is *fully* observable increment every T/Ntime
        - A int "last user action" wich is *fully* observable
        - A int "last agent action" wich is *fully* observable
        - The "user emotions" wich is "obscured"


         A single CreabotState represents a
    "guess" of the true belief state - which is the probability distribution over all states

    """

    def __init__(self, mood,last_strat,time_zone,da,idea_score,emotion=None):

        self.emotion = emotion #object of type Emotion
        self.mood = mood #object of type mood
        self.last_strat = last_strat
        self.time_zone = time_zone #int between 0 and NTime
        self.da = da # object of type AgentDa
        self.idea_score = idea_score #Object of type Idea

    def possible_next_state(self,action,next_time):
        if (self.emotion is not None):
            print("errrrorrr")
        else :

            states = []
            for m in range(N_MOOD):
                #for next_da in self.da.possible_next_da():
                for score in self.da.possible_idea_quality():
                    states.append(CreabotState(Mood(m),action.bin_number,next_time,AgentDa(O),Idea(score)))

        return states
    def as_tuple(self):
        return(self.last_strat,self.time_zone,self.idea_score.quality)




    def copy(self):

            return CreabotState(self.mood,self.last_strat,self.time_zone,self.da,self.idea_score,emotion = self.emotion)



    def to_string(self):
        state_string = ""
        state_string += self.mood.to_string()
        state_string += " - "
        state_string += str(self.time_zone)
        state_string += " - "
        state_string += str(Action(self.last_strat).to_string())
        state_string += " - "
        state_string += str(self.idea_score.to_string())
        state_string += " - "
        if self.emotion is not None:
            state_string += str(self.emotion.to_string())
            state_string += " - "


        return state_string

    def print(self):
        """
        Pretty printing
        :return:
        """
        print(self.to_string())
