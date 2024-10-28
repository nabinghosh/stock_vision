from enum import Enum

class ActionCode(Enum):
    Buy = 0
    Hold = 1
    Sell = 2


class ActionStatus(Enum):
    Success = 0
    Failed = -1


class Trader(object):
    def buy():
        pass
    def hold():
        pass
    def sell():
        pass