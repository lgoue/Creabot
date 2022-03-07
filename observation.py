from __future__ import print_function
from builtins import str


class Observation():
    """
    Default behavior is for the rock observation to say that the rock is empty
    """
    def __init__(self, P,A,D):
        self.D = D
        self.A = A
        self.P = P


    def distance_to(self, other_observation):
        return abs(self.P - other_observation.P)+ abs(self.A - other_observation.A) +abs(self.D - other_observation.D)

    def copy(self):
        return Observation(self.P,self.A,self.D)

    def __eq__(self, other_observation):
        return self.PAD == other_observation.PAD

    def __hash__(self):
        return self.PAD

    def print_observation(self):
        print("Measured PAD = (" + str(self.P) +"," + str(self.A)+ "," + str(self.D)+")")


    def to_string(self):

        obs = "Measured pleasure is" + str(self.P) +" measured arousal is" + str(self.A)+ "Measured dominance is" + str(self.D)
        return obs
