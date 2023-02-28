from dataclasses import dataclass


@dataclass
class FreqEval:
    """An evaluation stage for tracing learning curves."""

    pass


@dataclass
class LSEval:
    """An evaluation stage in the middle of a phase, where the agent is also checkpointed."""

    pass


@dataclass
class FinalEval:
    """An evaluation stage at the end of a phase, where data is collected for choosing the best agent of the phase.

    Args:
        i: Index of this final evaluation stage. Should start at 1.
    """

    i: int


EvalStage = FreqEval | LSEval | FinalEval
