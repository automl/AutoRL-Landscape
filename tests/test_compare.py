import numpy as np

from autorl_landscape.util.compare import choose_best_conf


def test_compare():
    run_ids = np.array([["a", "b", "c", "d"], ["e", "f", "g", "h"], ["i", "j", "k", "l"]])
    final_mean_rewards = np.array([[20, 0, 0, 1], [5, 6, 7, 8], [9, 11, 10, 8]])
    best_id = choose_best_conf(run_ids, final_mean_rewards, None)
    assert best_id == "j"
