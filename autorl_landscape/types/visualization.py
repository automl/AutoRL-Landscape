from typing import Any

from dataclasses import dataclass

from pandas import DataFrame


@dataclass
class Visualization:
    """Saves information for a plot."""

    title: str
    """Title for the plot (Visualizations with matching titles are drawn on the same `Axes`)"""
    viz_type: str
    """scatter, trisurf, etc."""
    viz_group: str
    """For allocating a Visualization to an image (combination of Visualizations)"""
    # x_samples: NDArray[Any]
    # y_samples: NDArray[Any]
    xy_norm: DataFrame
    """DataFrame including y (output) values for some visualization. May omit x values to assume the default, model.x"""
    # label: str
    kwargs: dict[str, Any]
