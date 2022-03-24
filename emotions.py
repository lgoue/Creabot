N_EMOTION = 25


class EmotionType(object):
    """
    Enumerates the OCC emotions
    """

    ADMIRATION = 0
    ANGER = 1
    DISLIKING = 2
    DISAPPOINTMENT = 3
    DISTRESS = 4
    FEAR = 5
    FEARSCONFIRMED = 6
    GLOATING = 7
    GRATIFICATION = 8
    GRATITUDE = 9
    HAPPYFOR = 10
    HATE = 11
    HOPE = 12
    JOY = 13
    LIKING = 14
    LOVE = 15
    PITY = 16
    PRIDE = 17
    RELIEF = 18
    REMORSE = 19
    REPROACH = 20
    RESENTMENT = 21
    SATISFACTION = 22
    SHAME = 23
    NEUTRAL = 24


class Emotion:
    def __init__(self, type):

        self.bin_number = type

    def distance_to_observation(self, observation):
        p, a, d = self.PAD_values()
        return (
            (observation.P - p) ** 2
            + (observation.A - a) ** 2
            + (observation.D - d) ** 2
        )

    def distance_to(self, emotion):
        if isinstance(emotion, int):
            emotion = Emotion(emotion)
        p, a, d = self.PAD_values()
        pe, ae, de = emotion.PAD_values()
        return (pe - p) ** 2 + (ae - a) ** 2 + (de - d) ** 2

    def PAD_values(self):
        p, a, d = "Error", "Error", "Error"
        match self.bin_number:
            case EmotionType.ADMIRATION:
                p, a, d = 0.5, 0.3, -0.2
            case EmotionType.ANGER:
                p, a, d = -0.51, 0.59, 0.25
            case EmotionType.DISLIKING:
                p, a, d = -0.4, 0.2, 0.1
            case EmotionType.DISAPPOINTMENT:
                p, a, d = -0.3, 0.1, -0.4
            case EmotionType.DISTRESS:
                p, a, d = -0.4, -0.2, -0.5
            case EmotionType.FEAR:
                p, a, d = -0.64, 0.60, -0.43
            case EmotionType.FEARSCONFIRMED:
                p, a, d = -0.5, -0.3, -0.7
            case EmotionType.GLOATING:
                p, a, d = 0.3, -0.3, -0.1
            case EmotionType.GRATIFICATION:
                p, a, d = 0.6, 0.5, 0.4
            case EmotionType.GRATITUDE:
                p, a, d = 0.4, 0.2, -0.3
            case EmotionType.HAPPYFOR:
                p, a, d = 0.4, 0.2, 0.2
            case EmotionType.HATE:
                p, a, d = -0.6, 0.6, 0.3
            case EmotionType.HOPE:
                p, a, d = 0.2, 0.2, -0.1
            case EmotionType.JOY:
                p, a, d = 0.4, 0.2, 0.1
            case EmotionType.LIKING:
                p, a, d = 0.40, 0.16, -0.24
            case EmotionType.LOVE:
                p, a, d = 0.3, 0.1, 0.2
            case EmotionType.PITY:
                p, a, d = -0.4, -0.2, -0.5
            case EmotionType.PRIDE:
                p, a, d = 0.4, 0.3, 0.3
            case EmotionType.RELIEF:
                p, a, d = 0.2, -0.3, 0.4
            case EmotionType.REMORSE:
                p, a, d = --0.3, 0.1, -0.6
            case EmotionType.REPROACH:
                p, a, d = -0.3, -0.1, 0.4
            case EmotionType.RESENTMENT:
                p, a, d = -0.2, -0.3, -0.2
            case EmotionType.SATISFACTION:
                p, a, d = 0.3, -0.2, 0.4
            case EmotionType.SHAME:
                p, a, d = -0.3, 0.1, -0.6
            case EmotionType.NEUTRAL:
                p, a, d = 0, 0, 0
        return p, a, d

    def to_string(self):
        e = "Unknown"
        match self.bin_number:
            case EmotionType.ADMIRATION:
                e = "Admiration"
            case EmotionType.ANGER:
                e = "Anger"
            case EmotionType.DISLIKING:
                e = "Disliking"
            case EmotionType.DISAPPOINTMENT:
                e = "Disappointment"
            case EmotionType.DISTRESS:
                e = "Distress"
            case EmotionType.FEAR:
                e = "Fear"
            case EmotionType.FEARSCONFIRMED:
                e = "FearsConfirmed"
            case EmotionType.GLOATING:
                e = "Gloating"
            case EmotionType.GRATIFICATION:
                e = "Gratification"
            case EmotionType.GRATITUDE:
                e = "Gratitude"
            case EmotionType.HAPPYFOR:
                e = "HappyFor"
            case EmotionType.HATE:
                e = "Hate"
            case EmotionType.HOPE:
                e = "Hope"
            case EmotionType.JOY:
                e = "Joy"
            case EmotionType.LIKING:
                e = "Liking"
            case EmotionType.LOVE:
                e = "Love"
            case EmotionType.PITY:
                e = "Pity"
            case EmotionType.PRIDE:
                e = "Pride"
            case EmotionType.RELIEF:
                e = "Relief"
            case EmotionType.REMORSE:
                e = "Remorse"
            case EmotionType.REPROACH:
                e = "Reproach"
            case EmotionType.RESENTMENT:
                e = "Resentment"
            case EmotionType.SATISFACTION:
                e = "Satisfaction"
            case EmotionType.SHAME:
                e = "Shame"
            case EmotionType.NEUTRAL:
                e = "Neutral"
        return e

        def print(self):
            print(self.to_string())
