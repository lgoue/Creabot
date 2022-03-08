from utils import *
from action import ActionType
from emotions import N_EMOTION,EmotionType, Emotion
import numpy as np
from Script import DialogActUser, IdeaQuality, Idea

EPS = 0.1
class User():

    def __init__(self):

        self.Extraversion = np.random.rand()
        self.Agreeableness =np.random.rand()
        self.Neuroticism =np.random.rand()
        self.Openness =np.random.rand()
        self.Conscientiousness = np.random.rand()
        self.mood = np.array([0.21*self.Extraversion + 0.59*self.Agreeableness + 0.19*self.Neuroticism ,
                         0.15*self.Openness + 0.30*self.Agreeableness - 0.57*self.Neuroticism,
                        0.25*self.Openness + 0.17*self.Conscientiousness + 0.60*self.Extraversion - 0.32*self.Agreeableness])
        self.Es = np.zeros(N_EMOTION)
        self.Ea = np.zeros(N_EMOTION)


        self.Mooddecay = 0.3
        self.Esdecay = 0.1

        self.da = DialogActUser.SILENCE

        self.emotions = [Emotion(e) for e in range(N_EMOTION)]
        self.alpha = [ list(e.PAD_values()) for e in self.emotions]

        self.last_idea_level= None

    def copy(self):
        user = User()
        user.Extraversion = self.Extraversion
        user.Agreeableness = self.Agreeableness
        user.Neuroticism = self.Neuroticism
        user.Openness = self.Openness
        user.Conscientiousness  = self.Conscientiousness
        user.mood = self.mood
        user.Es = self.Es

        return user

    def get_idea_quality(self):
        match self.da:
            case DialogActUser.NO_IDEA:
                last_idea_score = Idea(IdeaQuality.NO)
            case DialogActUser.GOOD_IDEA:
                last_idea_score = Idea(IdeaQuality.GOOD)
            case DialogActUser.MEDIUM_IDEA:
                last_idea_score = Idea(IdeaQuality.MEDIUM)
            case DialogActUser.BAD_IDEA:
                last_idea_score = Idea(IdeaQuality.BAD)
            case DialogActUser.DO_NOT_KNOW:
                last_idea_score = Idea(IdeaQuality.NO)
            case DialogActUser.ELLABORATE:
                last_idea_score = Idea(IdeaQuality.GOOD)
            case DialogActUser.GREET:
                last_idea_score = Idea(IdeaQuality.NA)
            case DialogActUser.SILENCE:
                last_idea_score = Idea(IdeaQuality.NO)
            case DialogActUser.YES:
                last_idea_score = Idea(IdeaQuality.NA)
            case DialogActUser.NO:
                last_idea_score = Idea(IdeaQuality.NA)
        return last_idea_score
    def updateEmotionalActivation(self, state, action):
        self.Ea = np.zeros(N_EMOTION)
        e = []
        match self.da:
            case DialogActUser.GOOD_IDEA:
                e.append(EmotionType.PRIDE)
                e.append(EmotionType.JOY)
                if state.last_strat == ActionType.ENCOURAGE:
                    e.append(EmotionType.SATISFACTION)
            case DialogActUser.NO_IDEA:
                e.append(EmotionType.SHAME)
                e.append(EmotionType.DISTRESS)
                if state.last_strat == ActionType.ENCOURAGE:
                    e.append(EmotionType.DISAPPOINTMENT)
            case DialogActUser.BAD_IDEA:
                e.append(EmotionType.FEAR)
            case DialogActUser.MEDIUM_IDEA:
                e.append(EmotionType.HOPE)
            case DialogActUser.ELLABORATE:
                e.append(EmotionType.JOY)
                if state.last_strat == ActionType.ENCOURAGE:
                    e.append(EmotionType.SATISFACTION)
            case DialogActUser.DO_NOT_KNOW:
                e.append(EmotionType.DISTRESS)
                if state.last_strat == ActionType.ENCOURAGE:
                    e.append(EmotionType.DISAPPOINTMENT)
        match action.bin_number:
            case ActionType.CRITICIZE:
                match self.last_idea_level:
                    case IdeaQuality.BAD:
                        e.append(EmotionType.FEARSCONFIRMED)
                        e.append(EmotionType.REMORSE)
                    case IdeaQuality.MEDIUM:
                        e.append(EmotionType.DISAPPOINTMENT)
                    case IdeaQuality.GOOD:
                        e.append(EmotionType.ANGER)
            case ActionType.PRAISE:
                match self.last_idea_level:
                    case IdeaQuality.BAD:
                        e.append(EmotionType.RELIEF)
                        e.append(EmotionType.GRATITUDE)
                    case IdeaQuality.MEDIUM:
                        e.append(EmotionType.SATISFACTION)
                    case IdeaQuality.GOOD:
                        e.append(EmotionType.GRATIFICATION)
            case ActionType.SELF_DISCLOSE:
                if self.goodmood():
                    e.append(EmotionType.ADMIRATION)
                else:
                    e.append(EmotionType.REPROACH)
            case ActionType.NEUTRAL:
                e.append(EmotionType.NEUTRAL)
            case ActionType.ENCOURAGE:
                e.append(EmotionType.HOPE)
        for emotion in e:
            self.Ea[emotion] = 1/len(e)
    def update_da(self, possible_da):
        match self.da:
            case DialogActUser.BAD_IDEA:
                self.last_idea_level = IdeaQuality.BAD
            case DialogActUser.MEDIUM_IDEA:
                self.last_idea_level = IdeaQuality.MEDIUM
            case DialogActUser.GOOD_IDEA:
                self.last_idea_level = IdeaQuality.GOOD

        n = len(possible_da)
        if n== 4:
            if self.mood[0]> 0.2:
                self.da = possible_da[3]
            elif self.mood[0]> 0:
                self.da = possible_da[2]
            elif self.mood[0]< -0.2:
                self.da = possible_da[0]
            else:
                self.da = possible_da[1]
        else :
            self.da= possible_da[np.random.randint(n)]
    def updateEmotionalState(self):

        Me = np.zeros(N_EMOTION)
        self.Es = self.Esdecay*self.Es
        for i in range(N_EMOTION):
            cm = int(self.alpha[i][1]*self.mood[1] > 0)
            Me[i] = np.abs(self.alpha[i][1]/2)*(1+cm*np.abs(self.mood[1]))

        #self.Es = self.Es + np.multiply(self.Ea,Me)
        self.Es = self.Es + self.Ea
    def updateMood(self):
        self.mood = self.Mooddecay*self.mood

        self.mood =  self.mood + np.dot(self.Es,self.alpha)

    def print_emotion(self):
        print("The user is in emotion : ",self.emotions[np.argmax(self.Es)].to_string(), "with intensity ",np.max(self.Es))
    def print_mood(self):
        print("The user is in mood : ",self.mood)

    def goodmood(self):
        return self.mood[0]>0.2


    def get_P(self):
        return self.mood[0] + np.random.rand()*EPS
    def get_A(self):
        return self.mood[1] + np.random.rand()*EPS
    def get_D(self):
        return self.mood[2] + np.random.rand()*EPS
