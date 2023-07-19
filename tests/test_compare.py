import numpy as np
import pytest

from autorl_landscape.run.compare import choose_best_policy, construct_2d


def test_compare() -> None:
    """Test comparing runs from multiple configurations, with multiple seeds.

    The final chosen run should come from the set of runs with the configuration that yields the best IQM. From this
    set, it should again be the run with the best IQM.
    """
    run_ids = np.array(
        [
            ["a", "b"],
            ["c", "d"],
            ["e", "f"],
            ["g", "h"],
            ["i", "j"],
            ["k", "l"],
        ],
    )
    # mean thinks that row 2 is best, chooses conf 2 from there -> "d"
    # IQM thinks that row 3 is best, chooses conf 1 from there -> "e"
    final_mean_rewards = np.array(
        [
            [[20, 0, 0, 1], [5, 6, 7, 8]],  # m 5.25, 6.5 (total 5.875); iqm 0.5 6.5 (total 4.75)
            [[9, 11, 10, 8], [100, 0, 0, 0]],  # m 9.5 25 (total 17.25); iqm 9.5 0 (total 6.75)
            [[8, 8, 8, 8], [7, 7, 7, 50]],  # m 8 17.75 (total 12.875); iqm 8 7 (total 7.75)
            [[np.nan, 1000, 1000, 1000], [1000, 1000, 1000, 1000]],  # existence of nan signifies crashed run, ignore
            [[np.nan, np.nan, np.nan, np.nan], [1000, 1000, 1000, 1000]],
            [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
        ]
    )
    best_id = choose_best_policy(run_ids, final_mean_rewards, None)
    assert best_id == "e"


def test_compare_all_crashed() -> None:
    """When all runs crash at least once, no optimal configuration can be picked."""
    run_ids = np.array([["a", "b"], ["c", "d"]])
    final_mean_rewards = np.array(
        [
            [[np.nan, 1], [123, 98576]],
            [[np.nan, np.nan], [np.nan, np.nan]],
        ]
    )
    with pytest.raises(ValueError, match="Cannot pick best configuration."):
        choose_best_policy(run_ids, final_mean_rewards, None)


def test_construct_2d() -> None:
    """Test that constructing the 2d array does not change the order of elements."""
    indices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    arr = np.arange(12)
    ret = construct_2d(indices, arr)
    assert np.all(ret == arr.reshape(4, 3))


def test_weird_data() -> None:
    """Test constructing the 2d array with expected datatypes."""
    # run_ids = np.full((len(conf.seeds),), "", dtype=np.dtype("<U8"))
    indices = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    ids = np.array([str(i) * 8 for i in range(8)], dtype=np.dtype("<U8"))
    data = np.arange(24)
    print(ids)
    ret_ids, ret_data = construct_2d(indices, ids, data.reshape(8, 3))
    assert np.array_equal(ret_ids, ids.reshape(4, 2))
    print(data.reshape(4, 2, 3))
    assert np.array_equal(ret_data, data.reshape(4, 2, 3))
