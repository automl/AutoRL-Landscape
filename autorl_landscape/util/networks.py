import haiku as hk
import jax


def neural_net(out_size: int, length: int, width: int) -> hk.Sequential:
    # net = hk.Sequential([hk.Linear(width), jax.nn.relu] * (length - 1) + [hk.Linear(out_size)])
    layers = []
    for _ in range(length - 1):
        layers += [hk.Linear(width), jax.nn.relu]
    layers += [hk.Linear(out_size)]
    return hk.Sequential(layers)
