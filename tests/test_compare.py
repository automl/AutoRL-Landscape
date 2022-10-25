import numpy as np

from autorl_landscape.util.compare import choose_best_conf, construct_2d


def test_compare():
    # run_ids = np.array([["a", "b", "c", "d"], ["e", "f", "g", "h"], ["i", "j", "k", "l"]])
    run_ids = np.array([["a", "b"], ["c", "d"], ["e", "f"]])
    # mean thinks that row 2 is best, chooses conf 2 from there -> "d"
    # IQM thinks that row 3 is best, chooses conf 1 from there -> "e"
    final_mean_rewards = np.array(
        [
            [[20, 0, 0, 1], [5, 6, 7, 8]],  # m 5.25, 6.5 (total 5.875); iqm 0.5 6.5 (total 4.75)
            [[9, 11, 10, 8], [100, 0, 0, 0]],  # m 9.5 25 (total 17.25); iqm 9.5 0 (total 6.75)
            [[8, 8, 8, 8], [7, 7, 7, 50]],  # m 8 17.75 (total 12.875); iqm 8 7 (total 7.75)
        ]
    )
    best_id = choose_best_conf(run_ids, final_mean_rewards, None)
    assert best_id == "e"


def test_construct_2d():
    indices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    arr = np.arange(12)
    ret = construct_2d(indices, arr)
    assert np.all(ret == arr.reshape(4, 3))


def test_weird_data():
    # run_ids = np.full((len(conf.seeds),), "", dtype=np.dtype("<U8"))
    indices = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    ids = np.array([str(i) * 8 for i in range(8)], dtype=np.dtype("<U8"))
    data = np.arange(24)
    print(ids)
    ret_ids, ret_data = construct_2d(indices, ids, data.reshape(8, 3))
    assert np.array_equal(ret_ids, ids.reshape(4, 2))
    print(data.reshape(4, 2, 3))
    assert np.array_equal(ret_data, data.reshape(4, 2, 3))
