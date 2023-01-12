from typing import Any

from collections import Counter

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from autorl_landscape.analyze.peaks import find_peaks
from autorl_landscape.ls_models.ls_model import LSModel, Visualization

BINS = 10


def check_modality(model: LSModel) -> None:
    """Count modes of per-configuration return distributions."""
    # min_counts: list[int] = []
    max_counts: list[int] = []
    modess_x: list[NDArray[Any]] = []
    modess_y: list[NDArray[Any]] = []
    ftu_PHIs = np.zeros((model.y.shape[0], 1))
    num_modes = np.zeros((model.y.shape[0], 1))
    for i, (x, y) in enumerate(zip(model.x, model.y)):
        # count peaks in histogram of distribution:
        y_hist, _ = np.histogram(y, BINS, range=(0, 1))
        _, max_mask, _, max_count = find_peaks(y_hist, 1)  # TODO change to KDE analysis
        # min_counts.append(min_count)
        max_counts.append(max_count)
        num_modes[i] = max_count

        # extract modes from histogram:
        modes_x = np.stack([x] * int(np.sum(max_mask)))
        modes_y = np.array((np.where(max_mask)[0] + 0.5) / BINS).reshape(-1, 1)
        modess_x.append(modes_x)
        modess_y.append(modes_y)

        # use FTU to classify distribution:
        from pyfolding import FTU  # lazy import

        ftu = FTU(y, routine="c++")
        ftu_PHIs[i] = ftu.folding_statistics  # FTU indicator for uni-modality; > 1 indicates uni-modality

    # min_counter = Counter(min_counts)
    max_counter = Counter(max_counts)
    # print(f"Minima in histogram: {sorted(min_counter.items())}")
    print(f"Maxima in histogram: {sorted(max_counter.items())}")

    model.add_viz_info(
        Visualization(
            "Folding Test of Uni-modality",
            "trisurf",
            "peaks",
            DataFrame(ftu_PHIs, columns=["Φ"]),
            {"cmap": "viridis"},
        )
    )
    model.add_viz_info(
        Visualization(
            "Per-Configuration Mode Count",
            "trisurf",
            "peaks",
            DataFrame(num_modes, columns=["mode count"]),
            {"cmap": "viridis"},
        )
    )
    modess_x = np.concatenate(modess_x)
    modess_y = np.concatenate(modess_y)
    model.add_viz_info(
        Visualization(
            "Per-Configuration Modes",
            "scatter",
            "graphs",
            model.build_df(modess_x, modess_y, "modes"),
            {"color": "red", "alpha": 0.75},
        )
    )
