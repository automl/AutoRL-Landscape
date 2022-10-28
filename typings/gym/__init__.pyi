import abc
from typing import Any

class Env(metaclass=abc.ABCMeta):
    def seed(self, seed: int) -> None: ...

class Wrapper(Env): ...

# from gym.envs import make
def make(id: str, **kwargs: Any) -> Env: ...
