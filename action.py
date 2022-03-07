from __future__ import print_function
from builtins import object
from emotions import EmotionType


N_ACTION = 5
class ActionType(object):
    """
    Enumerates the potential CreabotActions
    """
    CRITICIZE = 0
    PRAISE = 1
    ENCOURAGE = 2
    SELF_DISCLOSE = 3
    NEUTRAL = 4


class Action():
    def __init__(self, action_type):
        self.bin_number = action_type

    def copy(self):
        return CreabotAction(self.bin_number)

    def distance_to():
        pass

    def type_is(self,actiontype):
        return self.bin_number == actiontype

    def occ_emotion(self,state):
        e = "Unknown"
        match self.bin_number:
            case ActionType.CRITICIZE:
                e = EmotionType.ANGER
            case ActionType.PRAISE:
                e = EmotionType.PRIDE
            case ActionType.ENCOURAGE:
                e = EmotionType.GRATITUDE
            case ActionType.SELF_DISCLOSE:
                e = EmotionType.DISLIKING
            case ActionType.NEUTRAL:
                e = EmotionType.DISAPPOINTMENT
        return e

    def to_string(self):
        action="Unknown"
        if self.bin_number is ActionType.CRITICIZE:
            action = "Critic"
        elif self.bin_number is ActionType.PRAISE:
            action = "Praise"
        elif self.bin_number is ActionType.ENCOURAGE:
            action = "Encourage"
        elif self.bin_number is ActionType.SELF_DISCLOSE:
            action = "SD"
        elif self.bin_number is ActionType.NEUTRAL:
            action = "Neutral"
        return action

    def print_action(self):
        if self.bin_number is ActionType.CRITICIZE:
            print("Criticizing")
        elif self.bin_number is ActionType.PRAISE:
            print("Praising")
        elif self.bin_number is ActionType.ENCOURAGE:
            print("Encouraging")
        elif self.bin_number is ActionType.SELF_DISCLOSE:
            print("Self disclosing")
        elif self.bin_number is ActionType.NEUTRAL:
            print("Being Neutral")
