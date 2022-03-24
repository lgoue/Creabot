N_MOOD = 9


class MoodType(object):
    """
    Enumerates the OCC emotions
    """

    EXUBERANT = 0
    BORED = 1
    DEPENDENT = 2
    DISDAINFUL = 3
    RELAXED = 4
    ANXIOUS = 5
    DOCILE = 6
    HOSTILE = 7
    NEUTRAL = 8


class Mood:
    def __init__(self, type):

        self.bin_number = type

    def distance_to_observation(self, observation):
        p, a, d = self.PAD_values()
        return (
            (observation.P - p) ** 2
            + (observation.A - a) ** 2
            + (observation.D - d) ** 2
        )

    def distance_to(self, mood):
        if isinstance(mood, int):
            mood = Mood(mood)
        p, a, d = self.PAD_values()
        pe, ae, de = mood.PAD_values()
        return (pe - p) ** 2 + (ae - a) ** 2 + (de - d) ** 2

    def PAD_values(self):
        p, a, d = "Error", "Error", "Error"
        match self.bin_number:
            case MoodType.EXUBERANT:
                p, a, d = 0.5, 0.5, 0.5
            case MoodType.BORED:
                p, a, d = -0.5, -0.5, -0.5
            case MoodType.DEPENDENT:
                p, a, d = 0.5, 0.5, -0.5
            case MoodType.DISDAINFUL:
                p, a, d = -0.5, -0.5, 0.5
            case MoodType.RELAXED:
                p, a, d = 0.5, -0.5, 0.5
            case MoodType.ANXIOUS:
                p, a, d = -0.5, 0.5, -0.5
            case MoodType.DOCILE:
                p, a, d = 0.5, -0.5, -0.5
            case MoodType.HOSTILE:
                p, a, d = -0.5, 0.5, 0.5
            case MoodType.NEUTRAL:
                p, a, d = 0, 0, 0
        return p, a, d

    def to_string(self):
        e = "Unknown"
        match self.bin_number:
            case MoodType.EXUBERANT:
                e = "Exuberant"
            case MoodType.BORED:
                e = "Bored"
            case MoodType.DEPENDENT:
                e = "Dependent"
            case MoodType.DISDAINFUL:
                e = "Disdainful"
            case MoodType.RELAXED:
                e = "Relaxed"
            case MoodType.ANXIOUS:
                e = "Anxious"
            case MoodType.DOCILE:
                e = "Docile"
            case MoodType.HOSTILE:
                e = "Hostile"
            case MoodType.NEUTRAL:
                e = "Neutral"
        return e

        def print(self):
            print(self.to_string())
