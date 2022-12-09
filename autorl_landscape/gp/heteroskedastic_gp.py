from typing import Any

from pathlib import Path

import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import SVGP
from matplotlib.figure import Figure
from numpy.typing import NDArray

from autorl_landscape.util.data import read_wandb_csv
from autorl_landscape.util.ls_sampler import invert_ls_log_scaling

DTYPE = np.float64
gpf.config.set_default_float(DTYPE)

Data = tuple[NDArray[Any], ...]

LOWER = 0
UPPER = 500
INDUCE_GRID_LENGTH = 10
EPOCHS = 500


def grid_space_2d(length: int) -> NDArray[Any]:
    """Make grid in 2d unit cube. Returned array has shape (length ** 2, 2).

    Args:
        length: number of points on one side of the grid, aka. sqrt of the number of points returned.
    """
    axis = np.linspace(0, 1, num=length, dtype=DTYPE)
    grid_x, grid_y = np.meshgrid(axis, axis)
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    return np.stack((grid_x, grid_y), axis=1)


# https://gpflow.github.io/GPflow/develop/notebooks/advanced/heteroskedastic.html
def train_or_load_gps(phase_str: str, file: str, force_retrain: bool, save: bool) -> SVGP:
    """Train or load a GP model from disk.

    Args:
        phase_str: Which phase does the data belong to, which should be used to train or load the GP?
        file: csv data file.
        force_retrain: Whether to re-train the GP even if the model already exists on disk.
        save: Whether to save the GP to disk after training. Does nothing when just loading the GP from disk.

    Returns:
        The GP model
    """
    df = read_wandb_csv(file)
    phase_data = df[df["meta.phase"] == phase_str].sort_values("meta.conf_index")
    df_file = Path(file)
    # gp file is a sibling to the df file
    gp_folder = df_file.parent / f"{df_file.stem}_gp_{phase_str}"

    # y = np.array(list(phase_data["ls_eval/returns"]))  # (num_confs * num_seeds, num_evals) = (256 * 5, 20)
    conf_groups = phase_data.groupby(["meta.conf_index", "ls.learning_rate", "ls.gamma"])
    y = np.array(list(conf_groups["ls_eval/returns"].sum()), dtype=DTYPE)  # sum does concat (256, 100)
    # GET ls.learning_rate, ls.gamma, DISCARD meta.conf_index:
    x = np.array(list(conf_groups.groups.keys()), dtype=DTYPE)[:, 1:]
    # scale to interval [0, 1], undoing the logarithmic biases:
    x[:, 0] = invert_ls_log_scaling(x[:, 0], 1000, 0.0001, 0.1)  # TODO get inversion params from data
    x[:, 1] = invert_ls_log_scaling(1 - x[:, 1], 5, 0.0001, 0.2)

    if gp_folder.exists() and not force_retrain:  # load model
        model: SVGP = tf.saved_model.load(gp_folder)
        # with open(gp_file, "rb") as f:
        #     model: Any  # type: ignore[no-any-unimported]
        #     orig_final_loss: float
        #     model, orig_final_loss = pickle.load(f)
    else:  # train model
        # model definition:
        likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
            # distribution_class=build_truncated_normal(low=LOWER, high=UPPER),
            distribution_class=tfp.distributions.Normal,
            scale_transform=tfp.bijectors.Exp(),  # only positive values
        )
        # SquaredExponential (RBF) kernels for location and scale:
        lengthscales = np.array([0.1, 0.1], dtype=DTYPE)
        kernel = gpf.kernels.SeparateIndependent(
            [
                # gpf.kernels.SquaredExponential(lengthscales=0.1),  # location
                # gpf.kernels.SquaredExponential(lengthscales=0.1),  # scale
                gpf.kernels.SquaredExponential(lengthscales=lengthscales),  # location
                gpf.kernels.SquaredExponential(lengthscales=lengthscales),  # scale
            ]
        )
        induce_grid = grid_space_2d(INDUCE_GRID_LENGTH)
        # induce_grid = np.linspace(0, 1, 20).reshape(-1, 1)
        inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
            [
                gpf.inducing_variables.InducingPoints(induce_grid),
                gpf.inducing_variables.InducingPoints(induce_grid),
            ]
        )
        model: SVGP = gpf.models.SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            num_latent_gps=likelihood.latent_dim,
        )

        model.compiled_predict_f = tf.function(
            lambda Xnew: model.predict_f(Xnew, full_cov=False),
            input_signature=[tf.TensorSpec(shape=[None, 2], dtype=DTYPE)],
        )
        model.compiled_predict_y = tf.function(
            lambda Xnew: model.predict_y(Xnew, full_cov=False),
            input_signature=[tf.TensorSpec(shape=[None, 2], dtype=DTYPE)],
        )
        model.compiled_predict_f_samples = tf.function(
            lambda Xnew, num_samples: model.predict_f_samples(Xnew, num_samples, full_cov=False),
            input_signature=[tf.TensorSpec(shape=[None, 2], dtype=DTYPE), tf.TensorSpec(shape=[None], dtype=np.int64)],
        )

        # N = 101
        # X = np.linspace(0, 1, N).reshape(-1, 1)
        # rng = np.random.default_rng(0)
        # Y = rng.normal(np.sin(10 * X), 0.5 * np.exp(np.cos(10 * X))).reshape(-1, 1)
        # model, orig_final_loss = _train_gp(model, (X, Y))

        # # toy data
        # n_samples = 3
        # X_ = grid_space_2d(8)
        # X = np.repeat(X_, n_samples, axis=0)
        # rng = np.random.default_rng(0)
        # # Y = rng.normal(np.sin(10 * X[:,0]) + 5 * X[:, 0] + 10, X[:,1]).reshape(-1, 1)
        # Y = rng.normal(5 * X[:, 0] + 10, X[:, 1]).reshape(-1, 1)
        # # Y = tfp.distributions.TruncatedNormal(0, 1, 0, 1).sample((1000, 1)).numpy()
        # # Y = truncnorm.rvs(a=LOWER, b=UPPER, size=(8 * 8 * 3, 1), loc=0, scale=2).astype(DTYPE)
        # # Y = rng.normal(5, X[:, 1] + 0.5 * X[:, 0]).reshape(-1, 1)
        # model, orig_final_loss = _train_gp(model, (X, Y), early_stopping=False, n_plots=0)

        # real data
        model, orig_final_loss = _train_gp(
            model, (np.repeat(x, y.shape[1], axis=0), y.reshape(-1, 1)), early_stopping=False
        )

        # model, orig_final_loss = _train_gp(model, (x, np.ones_like(y_means.reshape(-1,1))))
        # model, orig_final_loss = _train_gp(
        #     model, (x, np.random.normal(np.sin(50 * x[:, 0]), np.exp(np.cos(50 * x[:, 0]))).reshape(-1, 1))
        # )

        if save:
            tf.saved_model.save(model, gp_folder)
            # with open(gp_file, "wb") as f:
            #     pickle.dump((model, orig_final_loss), f)

    # TODO redo sanity check?
    # score = model.score(x, y_means)
    # assert math.isclose(score, orig_score), "R²-Score of loaded GPR differs from original score!"
    return model


def early_stop(last_k_losses: NDArray[Any], rel_tol: float = 0.001, verbose: bool = False) -> bool:
    """Early stopping based on ratio of max and min in an array."""
    upper = np.max(last_k_losses)
    lower = np.min(last_k_losses)
    r = upper / lower
    if r < 1 + rel_tol:
        if verbose:
            print(f"Stopping early after last {last_k_losses.size} checked performance values were close enough.")
            print(f"{r} < 1 + {rel_tol}")
        return True
    return False


def _train_gp(
    model: SVGP, data: tuple[NDArray[Any], NDArray[Any]], early_stopping: bool, n_plots: int = 0
) -> tuple[SVGP, float]:
    batch_size = 64
    dataset = tf.data.Dataset.from_tensor_slices(data)
    batched_dataset = dataset.shuffle(len(dataset), reshuffle_each_iteration=True).batch(batch_size)
    # loss_fn = model.training_loss_closure(iter(data))

    gpf.utilities.set_trainable(model.q_mu, False)
    gpf.utilities.set_trainable(model.q_sqrt, False)

    variational_vars = [(model.q_mu, model.q_sqrt)]
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

    adam_vars = model.trainable_variables
    # adam_opt = tf.optimizers.Adam(0.01)
    adam_opt = tf.optimizers.Adam()

    @tf.function
    def optimization_step(model: SVGP, batch: tuple[tf.Tensor, tf.Tensor]) -> float:  # one epoch of steps
        # for opt, vars in [(natgrad_opt, variational_vars), (adam_opt, adam_vars)]:
        #     with tf.GradientTape(watch_accessed_variables=False) as tape:
        #         tape.watch(vars)
        #         loss = model.training_loss(batch)
        #     grads = tape.gradient(loss, vars)
        #     opt.apply_gradients(zip(grads, vars))
        # return float(loss)
        loss_fn = model.training_loss_closure(batch)
        natgrad_opt.minimize(loss_fn, variational_vars)
        adam_opt.minimize(loss_fn, adam_vars)
        return loss_fn()

    if n_plots > 0:
        fig = plt.figure(figsize=(16, 12))
    epochs = EPOCHS
    n_logs = min(100, epochs)
    log_interval = epochs // n_logs
    plot_interval = epochs // n_plots if n_plots > 0 else 2 * epochs
    n = epochs // plot_interval
    final_loss: float
    i = 0  # log counter
    j = 1  # plot counter
    k = 5
    last_k_losses = np.zeros(k)

    for epoch in range(1, epochs + 1):
        for batch in batched_dataset:
            loss = optimization_step(model, batch)
        if epoch % log_interval == 0:
            # loss: float = loss_fn().numpy()
            final_loss = loss
            last_k_losses[i % k] = loss
            print(f"Epoch {epoch} - Loss: {loss:.4f}")
            print(f"loc kernel l's: {model.kernel.kernels[0].lengthscales}")
            print(f"std kernel l's: {model.kernel.kernels[1].lengthscales}")
            i += 1
        if epoch % plot_interval == 0:
            plot_gp(model, fig, n, j, data, n_plots)
            j += 1
        if early_stopping and i >= k and early_stop(last_k_losses, verbose=True):
            break
    if n_plots > 0:
        fig.legend()
        plt.show()
    return (model, final_loss)


def plot_gp(model: SVGP, fig: Figure, n: int, i: int, data: Data) -> None:
    """Plot a GP model and training data."""
    match len(data):
        case 2:
            return _plot_1d(model, fig, n, i, data)
        case 3:
            return _plot_2d(model, fig, n, i, data)
        case _:
            raise Exception("Cannot infer dimensionality of LS from given data.")


def _plot_1d(model: SVGP, fig: Figure, n: int, i: int, data: Data) -> None:
    grid_size = 50
    grid = np.linspace(0, 1, num=grid_size).reshape(-1, 1)

    gp_mean, gp_var = model.compiled_predict_y(grid)
    gp_mean = gp_mean.numpy().flatten()
    print(np.std(gp_mean))
    gp_std = np.sqrt(gp_var.numpy().flatten())

    ax = fig.add_subplot(1, n, i)
    ax.plot(grid, gp_mean, color="black", label="mean")
    ax.fill_between(grid.flatten(), gp_mean - 1.96 * gp_std, gp_mean + 1.96 * gp_std, label="CI", alpha=0.25)
    if data is not None:
        ax.scatter(data[0], data[1], label="samples")
    fig.legend()
    # for y_, opacity in (
    #     (gp_mean, 1.0),
    #     (gp_mean + 1.96 * gp_std, 0.5),
    #     (gp_mean - 1.96 * gp_std, 0.5),
    # ):
    #     cmap = "viridis" if opacity == 1.0 else None
    #     color = (0.5, 0.5, 0.5, opacity) if opacity < 1.0 else None
    #     ax.plot(grid, y_, color=color)


def _plot_2d(model: SVGP, fig: Figure, n: int, i: int, data: Data) -> None:
    grid_size = 50
    grid_x1, grid_x2 = np.meshgrid(np.linspace(0, 1, num=grid_size), np.linspace(0, 1, num=grid_size))
    grid_x1 = grid_x1.flatten()
    grid_x2 = grid_x2.flatten()
    grid = np.stack((grid_x1, grid_x2), axis=1)

    # # samples
    # n_samples = 1
    # sampled_fs = model.predict_f_samples(grid, n_samples)

    gp_mean, gp_var = model.compiled_predict_y(grid)
    gp_mean = gp_mean.numpy().flatten()
    gp_std = np.sqrt(gp_var.numpy().flatten())

    ax = fig.add_subplot(1, n, i, projection="3d")
    ax.set_zlim3d(LOWER, UPPER)

    # # samples
    # ax.plot_surface(
    #     grid_x1.reshape(grid_size, grid_size),
    #     grid_x2.reshape(grid_size, grid_size),
    #     sampled_fs[:, :, 0].numpy().reshape(grid_size, grid_size),
    #     edgecolor="none",
    #     cmap=cmap,
    #     color=color,
    #     shade=False,
    # )

    for y_, opacity in (
        (gp_mean, 1.0),
        (gp_mean + 1.96 * gp_std, 0.5),
        (gp_mean - 1.96 * gp_std, 0.5),
    ):
        cmap = "viridis" if opacity == 1.0 else None
        color = (0.5, 0.5, 0.5, opacity) if opacity < 1.0 else None
        ax.plot_surface(
            grid_x1.reshape(grid_size, grid_size),
            grid_x2.reshape(grid_size, grid_size),
            y_.reshape(grid_size, grid_size),
            cmap=cmap,
            color=color,
            edgecolor="none",
            shade=False,
        )
    if data is not None:
        ax.scatter(data[0], data[1], data[2], label="samples")
