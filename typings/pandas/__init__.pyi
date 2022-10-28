from typing import Any, Hashable, Iterable, Optional, Union
from numpy.typing import ArrayLike
class Series:
    def __getitem__(self, key: str) -> Any: ...
class DataFrame:
    def __init__(
        self,
        data: ArrayLike,
        # index: Union[Axes, None] = ...,
        columns: Optional[ArrayLike] = ...,
        # dtype: Union[Dtype, None] = ...,
        # copy: Union[bool, None] = ...,
    ) -> None: ...
    def iterrows(self) -> Iterable[tuple[Hashable, Series]]: ...
    def __getitem__(self, key: Union[str, ArrayLike]) -> Any: ...
