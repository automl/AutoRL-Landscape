from typing import Any

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from KDEpy.FFTKDE import FFTKDE
from numpy.typing import NDArray

from autorl_landscape.ls_models.ls_model import LSModel, Visualization

KDE_GRID_LENGTH = 1000


def check_modality(model: LSModel, bw: float) -> None:
    """Count modes of per-configuration return distributions."""
    bw = 0.05
    # min_counts: list[int] = []
    max_counts: list[int] = []
    modess_x: list[NDArray[Any]] = []
    modess_y: list[NDArray[Any]] = []
    # num_modes = np.zeros((model.y.shape[0], 1))
    for i, (x, y) in enumerate(zip(model.x, model.y)):  # for every conf:
        kde_x = np.linspace(-0.01, 1.01, KDE_GRID_LENGTH)
        kde_y = FFTKDE(kernel="tri", bw=bw).fit(y).evaluate(kde_x)
        # kde_y = TruncatedNormalKDE(bw=bw, a=0, b=1).fit(y).evaluate(kde_x)

        fig = plt.figure(figsize=(16, 12))
        ax = fig.gca()
        ax.scatter(y, np.zeros_like(y))
        ax.plot(kde_x, kde_y)
        ax.set_xlim((-1, 2))
        plt.show()

        continue

        # max_count = len(intervals)
        # max_counts.append(max_count)
        # num_modes[i] = max_count

        # modes_x = np.stack([x] * max_count)
        # modes_y = np.array([np.mean(y[list(interval)]) for interval in intervals])
        # modess_x.append(modes_x)
        # modess_y.append(modes_y)

    # min_counter = Counter(min_counts)
    max_counter = Counter(max_counts)
    # print(f"Minima in histogram: {sorted(min_counter.items())}")
    print(f"Maxima in histogram: {sorted(max_counter.items())}")

    # model.add_viz_info(
    #     Visualization(
    #         "Per-Configuration Mode Count",
    #         "trisurf",
    #         "peaks",
    #         DataFrame(num_modes, columns=["mode count"]),
    #         {"cmap": "viridis"},
    #     )
    # )
    modess_x = np.concatenate(modess_x)
    modess_y = np.concatenate(modess_y)
    model.add_viz_info(
        Visualization(
            "Per-Configuration Modes",
            "scatter",
            "modalities",
            model.build_df(modess_x, modess_y, "modes"),
            {"color": "red", "alpha": 0.75},
        )
    )
