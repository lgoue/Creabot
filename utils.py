import numpy as np
def softmax(x) :
    return np.exp(x)/sum(np.exp(x))

def entropy(x):
    return x*np.log(x+0.00001)

IndextoMood = ["Exuberant" , "Bored","Dependent","Disdainful", "Relaxed", "Anxious", "Docile" ,"Hostile", "Neutral"]
MoodtoIndex = {"Exuberant":0, "Bored":1,"Dependent":2,"Disdainful":3, "Relaxed":4, "Anxious":5, "Docile":6 ,"Hostile":7, "Neutral":8}
GoodMoods = ["Exuberant","Relaxed","Docile"]
