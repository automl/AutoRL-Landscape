from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class FreqEval:
    """An evaluation stage for tracing learning curves."""

    num_episodes: int

    def log_name(self) -> str:
        """Return name of this evaluation type, to be used in wandb logging."""
        return "freq_eval"


@dataclass(eq=True, frozen=True)
class LSEval:
    """An evaluation stage in the middle of a phase, where the agent is also checkpointed."""

    num_episodes: int

    def log_name(self) -> str:
        """Return name of this evaluation type, to be used in wandb logging."""
        return "ls_eval"


@dataclass(eq=True, frozen=True)
class FinalEval:
    """An evaluation stage at the end of a phase, where data is collected for choosing the best agent of the phase.

    Args:
        i: Index of this final evaluation stage. Should start at 1.
    """

    num_episodes: int
    i: int

    def log_name(self) -> str:
        """Return name of this evaluation type, to be used in wandb logging."""
        return f"final_eval_{self.i}"


EvalStage = FreqEval | LSEval | FinalEval
