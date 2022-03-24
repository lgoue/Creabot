import numpy as np

N_IDEA_QUALITY = 5
N_AGENT_DA = 10


class IdeaQuality(object):
    NO = -2
    BAD = -1
    MEDIUM = 1
    GOOD = 2
    NA = 0


class Idea:
    def __init__(self, quality):
        self.quality = quality

    def to_string(self):
        match self.quality:
            case IdeaQuality.BAD:
                return "Bad Idea"
            case IdeaQuality.MEDIUM:
                return "MEDIUM_IDEA"
            case IdeaQuality.GOOD:
                return "Good Idea"
            case IdeaQuality.NO:
                return "No Idea"
            case IdeaQuality.NA:
                return "Idea quality not applicable"

    def print(self):
        print(self.to_string())


class DialogActAgent(object):
    """
    Enumerates the OCC emotions
    """

    GREETING = 0
    PROBLEM_STATEMENT = 1
    COMMENT_REPHRASE = 2
    ANY_IDEA = 3
    DRQ_NEW_THEME = 4
    ACKNOWLEDGE = 5
    LLQ = 6
    REPHRASE = 7
    DRQ = 8
    COMMENT_DO_NOT_KNOW = 9


class DialogActUser(object):
    """
    Enumerates the OCC emotions
    """

    GREET = 0
    YES = 1
    NO = 2
    DO_NOT_KNOW = 3
    ELLABORATE = 4
    GOOD_IDEA = 5
    BAD_IDEA = 6
    MEDIUM_IDEA = 7
    NO_IDEA = 8
    SILENCE = 9


class AgentDa:
    def __init__(self, bin):
        self.bin = bin

    def possible_idea_quality(self):
        match self.bin:
            case DialogActAgent.GREETING:
                return [DialogActUser.GREET]
            case DialogActAgent.PROBLEM_STATEMENT:
                return [
                    DialogActUser.NO_IDEA,
                    DialogActUser.GOOD_IDEA,
                    DialogActUser.BAD_IDEA,
                    DialogActUser.MEDIUM_IDEA,
                ]
            case DialogActAgent.ANY_IDEA:
                return [
                    DialogActUser.NO_IDEA,
                    DialogActUser.GOOD_IDEA,
                    DialogActUser.BAD_IDEA,
                    DialogActUser.MEDIUM_IDEA,
                ]
            case DialogActAgent.DRQ_NEW_THEME:
                return [
                    DialogActUser.NO_IDEA,
                    DialogActUser.GOOD_IDEA,
                    DialogActUser.BAD_IDEA,
                    DialogActUser.MEDIUM_IDEA,
                ]
            case DialogActAgent.ACKNOWLEDGE:
                return [
                    DialogActUser.NO_IDEA,
                    DialogActUser.GOOD_IDEA,
                    DialogActUser.BAD_IDEA,
                    DialogActUser.MEDIUM_IDEA,
                ]
            case DialogActAgent.LLQ:
                return [DialogActUser.ELLABORATE, DialogActUser.DO_NOT_KNOW]
            case DialogActAgent.REPHRASE:
                return [DialogActUser.YES, DialogActUser.NO]
            case DialogActAgent.DRQ:
                return [
                    DialogActUser.NO_IDEA,
                    DialogActUser.GOOD_IDEA,
                    DialogActUser.BAD_IDEA,
                    DialogActUser.MEDIUM_IDEA,
                ]
            case DialogActAgent.COMMENT_DO_NOT_KNOW:
                return [
                    DialogActUser.NO_IDEA,
                    DialogActUser.GOOD_IDEA,
                    DialogActUser.BAD_IDEA,
                    DialogActUser.MEDIUM_IDEA,
                ]
            case DialogActAgent.COMMENT_REPHRASE:
                return [
                    DialogActUser.NO_IDEA,
                    DialogActUser.GOOD_IDEA,
                    DialogActUser.BAD_IDEA,
                    DialogActUser.MEDIUM_IDEA,
                ]

        def update_state(self, user_da):

            match self.bin:
                case DialogActAgent.GREETING:
                    return [IdeaQuality.NA]

                case DialogActAgent.PROBLEM_STATEMENT:
                    return [
                        IdeaQuality.NO,
                        IdeaQuality.GOOD,
                        IdeaQuality.MEDIUM,
                        IdeaQuality.BAD,
                    ]
                case DialogActAgent.ANY_IDEA:
                    return [
                        IdeaQuality.NO,
                        IdeaQuality.GOOD,
                        IdeaQuality.MEDIUM,
                        IdeaQuality.BAD,
                    ]
                case DialogActAgent.DRQ_NEW_THEME:
                    return [
                        IdeaQuality.NO,
                        IdeaQuality.GOOD,
                        IdeaQuality.MEDIUM,
                        IdeaQuality.BAD,
                    ]
                case DialogActAgent.ACKNOWLEDGE:
                    return [
                        IdeaQuality.NO,
                        IdeaQuality.GOOD,
                        IdeaQuality.MEDIUM,
                        IdeaQuality.BAD,
                    ]
                case DialogActAgent.LLQ:
                    return [IdeaQuality.NO, IdeaQuality.GOOD]
                case DialogActAgent.REPHRASE:
                    return [IdeaQuality.NA]

                case DialogActAgent.DRQ:
                    return [
                        IdeaQuality.NO,
                        IdeaQuality.GOOD,
                        IdeaQuality.MEDIUM,
                        IdeaQuality.BAD,
                    ]
                case DialogActAgent.COMMENT_DO_NOT_KNOW:
                    return [
                        IdeaQuality.NO,
                        IdeaQuality.GOOD,
                        IdeaQuality.MEDIUM,
                        IdeaQuality.BAD,
                    ]
                case DialogActAgent.COMMENT_REPHRASE:
                    return [
                        IdeaQuality.NO,
                        IdeaQuality.GOOD,
                        IdeaQuality.MEDIUM,
                        IdeaQuality.BAD,
                    ]

    def possible_next_da(self):
        match self.bin:
            case DialogActAgent.GREETING:
                return [DialogActAgent.PROBLEM_STATEMENT]
            case DialogActAgent.PROBLEM_STATEMENT:
                return [
                    DialogActAgent.DRQ_NEW_THEME,
                    DialogActAgent.ACKNOWLEDGE,
                    DialogActAgent.LLQ,
                ]
            case DialogActAgent.ANY_IDEA:
                return [
                    DialogActAgent.DRQ_NEW_THEME,
                    DialogActAgent.ACKNOWLEDGE,
                    DialogActAgent.LLQ,
                ]
            case DialogActAgent.DRQ_NEW_THEME:
                return [
                    DialogActAgent.DRQ_NEW_THEME,
                    DialogActAgent.ACKNOWLEDGE,
                    DialogActAgent.LLQ,
                ]
            case DialogActAgent.ACKNOWLEDGE:
                return [
                    DialogActAgent.DRQ_NEW_THEME,
                    DialogActAgent.ACKNOWLEDGE,
                    DialogActAgent.LLQ,
                ]
            case DialogActAgent.LLQ:
                return [DialogActAgent.COMMENT_DO_NOT_KNOW, DialogActAgent.REPHRASE]
            case DialogActAgent.REPHRASE:
                return [DialogActAgent.COMMENT_REPHRASE, DialogActAgent.DRQ]
            case DialogActAgent.DRQ:
                return [
                    DialogActAgent.DRQ_NEW_THEME,
                    DialogActAgent.ACKNOWLEDGE,
                    DialogActAgent.LLQ,
                ]
            case DialogActAgent.COMMENT_DO_NOT_KNOW:
                return [
                    DialogActAgent.DRQ_NEW_THEME,
                    DialogActAgent.ACKNOWLEDGE,
                    DialogActAgent.LLQ,
                ]
            case DialogActAgent.COMMENT_REPHRASE:
                return [
                    DialogActAgent.DRQ_NEW_THEME,
                    DialogActAgent.ACKNOWLEDGE,
                    DialogActAgent.LLQ,
                ]

    def to_string(self):
        match self.bin:
            case DialogActAgent.GREETING:
                return "greet"
            case DialogActAgent.PROBLEM_STATEMENT:
                return "prob statement"
            case DialogActAgent.COMMENT_REPHRASE:
                return "comment rephrase"
            case DialogActAgent.ANY_IDEA:
                return "any idea"
            case DialogActAgent.DRQ_NEW_THEME:
                return "DRQ new theme"
            case DialogActAgent.ACKNOWLEDGE:
                return "acknowledge idea"
            case DialogActAgent.LLQ:
                return "llq"
            case DialogActAgent.REPHRASE:
                return "rephrase"
            case DialogActAgent.DRQ:
                return "drq"
            case DialogActAgent.COMMENT_DO_NOT_KNOW:
                return "a shame you do not know"

        def print(self):
            print(self.to_string())


class UserDa:
    def __init__(self, bin):
        self.bin = bin

    def to_string(self):
        match self.bin:
            case DialogActUser.GREET:
                return "greet"
            case DialogActUser.YES:
                return "yes"
            case DialogActUser.NO:
                return "no"
            case DialogActUser.DO_NOT_KNOW:
                return "don't know"
            case DialogActUser.ELLABORATE:
                return "ellaborate"
            case DialogActUser.GOOD_IDEA:
                return "good idea"
            case DialogActUser.BAD_IDEA:
                return "bad idea"
            case DialogActUser.MEDIUM_IDEA:
                return "medium idea"
            case DialogActUser.NO_IDEA:
                return "no idea"
            case DialogActUser.SILENCE:
                return "silence"

    def print(self):
        print(self.to_string())


class Script:
    def __init__(self):

        self.current_state = DialogActAgent.GREETING

    def get_user_possible_da(self):

        match self.current_state:
            case DialogActAgent.GREETING:
                return [DialogActUser.GREET]

            case DialogActAgent.PROBLEM_STATEMENT:
                return [
                    DialogActUser.NO_IDEA,
                    DialogActUser.BAD_IDEA,
                    DialogActUser.MEDIUM_IDEA,
                    DialogActUser.GOOD_IDEA,
                ]

            case DialogActAgent.ANY_IDEA:
                return [
                    DialogActUser.NO_IDEA,
                    DialogActUser.BAD_IDEA,
                    DialogActUser.MEDIUM_IDEA,
                    DialogActUser.GOOD_IDEA,
                ]
            case DialogActAgent.DRQ_NEW_THEME:
                return [
                    DialogActUser.NO_IDEA,
                    DialogActUser.BAD_IDEA,
                    DialogActUser.MEDIUM_IDEA,
                    DialogActUser.GOOD_IDEA,
                ]
            case DialogActAgent.ACKNOWLEDGE:
                return [
                    DialogActUser.NO_IDEA,
                    DialogActUser.BAD_IDEA,
                    DialogActUser.MEDIUM_IDEA,
                    DialogActUser.GOOD_IDEA,
                ]
            case DialogActAgent.LLQ:
                return [DialogActUser.ELLABORATE, DialogActUser.DO_NOT_KNOW]
            case DialogActAgent.REPHRASE:
                return [DialogActUser.YES, DialogActUser.NO]
            case DialogActAgent.DRQ:
                return [
                    DialogActUser.NO_IDEA,
                    DialogActUser.BAD_IDEA,
                    DialogActUser.MEDIUM_IDEA,
                    DialogActUser.GOOD_IDEA,
                ]
            case DialogActAgent.COMMENT_DO_NOT_KNOW:
                return [
                    DialogActUser.NO_IDEA,
                    DialogActUser.BAD_IDEA,
                    DialogActUser.MEDIUM_IDEA,
                    DialogActUser.GOOD_IDEA,
                ]
            case DialogActAgent.COMMENT_REPHRASE:
                return [
                    DialogActUser.NO_IDEA,
                    DialogActUser.BAD_IDEA,
                    DialogActUser.MEDIUM_IDEA,
                    DialogActUser.GOOD_IDEA,
                ]

    def update_state(self, user_da):

        match self.current_state:
            case DialogActAgent.GREETING:
                self.current_state = DialogActAgent.PROBLEM_STATEMENT
                match user_da:
                    case DialogActUser.GREET:
                        self.current_state = DialogActAgent.PROBLEM_STATEMENT

            case DialogActAgent.PROBLEM_STATEMENT:
                match user_da:
                    case DialogActUser.NO_IDEA:
                        self.current_state = DialogActAgent.DRQ_NEW_THEME
                    case DialogActUser.SILENCE:
                        self.current_state = DialogActAgent.DRQ_NEW_THEME

                    case DialogActUser.GOOD_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
                    case DialogActUser.BAD_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
                    case DialogActUser.MEDIUM_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
            case DialogActAgent.ANY_IDEA:
                match user_da:
                    case DialogActUser.NO_IDEA:
                        self.current_state = DialogActAgent.DRQ_NEW_THEME
                    case DialogActUser.SILENCE:
                        self.current_state = DialogActAgent.DRQ_NEW_THEME

                    case DialogActUser.GOOD_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
                    case DialogActUser.BAD_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
                    case DialogActUser.MEDIUM_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
            case DialogActAgent.DRQ_NEW_THEME:
                match user_da:
                    case DialogActUser.NO_IDEA:
                        self.current_state = DialogActAgent.DRQ_NEW_THEME
                    case DialogActUser.SILENCE:
                        self.current_state = DialogActAgent.DRQ_NEW_THEME

                    case DialogActUser.GOOD_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
                    case DialogActUser.BAD_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
                    case DialogActUser.MEDIUM_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
            case DialogActAgent.ACKNOWLEDGE:
                match user_da:
                    case DialogActUser.NO_IDEA:
                        self.current_state = DialogActAgent.DRQ_NEW_THEME
                    case DialogActUser.SILENCE:
                        self.current_state = DialogActAgent.DRQ_NEW_THEME

                    case DialogActUser.GOOD_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
                    case DialogActUser.BAD_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
                    case DialogActUser.MEDIUM_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
            case DialogActAgent.LLQ:
                self.current_state = DialogActAgent.COMMENT_DO_NOT_KNOW
                match user_da:
                    case DialogActUser.ELLABORATE:
                        self.current_state = DialogActAgent.REPHRASE
                    case DialogActUser.DO_NOT_KNOW:
                        self.current_state = DialogActAgent.COMMENT_DO_NOT_KNOW

            case DialogActAgent.REPHRASE:
                self.current_state = DialogActAgent.COMMENT_REPHRASE
                match user_da:
                    case DialogActUser.YES:
                        self.current_state = DialogActAgent.DRQ

            case DialogActAgent.DRQ:
                match user_da:
                    case DialogActUser.NO_IDEA:
                        self.current_state = DialogActAgent.DRQ_NEW_THEME
                    case DialogActUser.SILENCE:
                        self.current_state = DialogActAgent.DRQ_NEW_THEME

                    case DialogActUser.GOOD_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
                    case DialogActUser.BAD_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
                    case DialogActUser.MEDIUM_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
            case DialogActAgent.COMMENT_DO_NOT_KNOW:
                match user_da:
                    case DialogActUser.NO_IDEA:
                        self.current_state = DialogActAgent.DRQ_NEW_THEME
                    case DialogActUser.SILENCE:
                        self.current_state = DialogActAgent.DRQ_NEW_THEME

                    case DialogActUser.GOOD_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
                    case DialogActUser.BAD_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
                    case DialogActUser.MEDIUM_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
            case DialogActAgent.COMMENT_REPHRASE:
                match user_da:
                    case DialogActUser.NO_IDEA:
                        self.current_state = DialogActAgent.DRQ_NEW_THEME
                    case DialogActUser.SILENCE:
                        self.current_state = DialogActAgent.DRQ_NEW_THEME
                    case DialogActUser.GOOD_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
                    case DialogActUser.BAD_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
                    case DialogActUser.MEDIUM_IDEA:
                        p = np.random.rand()
                        if p > 0.5:
                            self.current_state = DialogActAgent.ACKNOWLEDGE
                        else:
                            self.current_state = DialogActAgent.LLQ
