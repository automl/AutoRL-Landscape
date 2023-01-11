from typing import Any

from pathlib import Path

import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import SVGP
from numpy.typing import NDArray
from pandas import DataFrame

from autorl_landscape.ls_models.ls_model import LSModel
from autorl_landscape.util.grid_space import grid_space_2d

# DTYPE = np.float64


LOWER = 0
UPPER = 500
# INDUCE_GRID_LENGTH = 10


class HSGPModel(LSModel):
    """Heteroskedastic GP Model."""

    def __init__(
        self,
        data: DataFrame,
        induce_grid_length: int,
        dtype: type = np.float64,
        y_col: str = "ls_eval/returns",
        y_bounds: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(data, dtype, y_col, y_bounds)
        self.induce_grid_length = induce_grid_length
        gpf.config.set_default_float(dtype)  # WARNING Global!

    def fit(self, epochs: int, batch_size: int = 64, early_stopping: bool = False, verbose: bool = False) -> None:
        """Fit the GP to its data.

        Args:
            epochs: Number of epochs to train for.
            batch_size: Batch size for training data.
            early_stopping: Whether to employ early stopping based on the last 5 epochs, based on relative difference
                between min and max.
            verbose: Verbosity.
        """
        # Define model:
        likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
            # distribution_class=build_truncated_normal(low=LOWER, high=UPPER),
            distribution_class=tfp.distributions.Normal,
            scale_transform=tfp.bijectors.Exp(),  # only positive values
        )
        # SquaredExponential (RBF) kernels for location and scale:
        lengthscales = np.array([0.1, 0.1], dtype=self.dtype)
        kernel = gpf.kernels.SeparateIndependent(
            [
                gpf.kernels.SquaredExponential(lengthscales=lengthscales),  # location
                gpf.kernels.SquaredExponential(lengthscales=lengthscales),  # scale
            ]
        )
        induce_grid = grid_space_2d(self.induce_grid_length, self.dtype)
        # induce_grid = np.linspace(0, 1, 20).reshape(-1, 1)
        inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
            [
                gpf.inducing_variables.InducingPoints(induce_grid),
                gpf.inducing_variables.InducingPoints(induce_grid),
            ]
        )
        self.model: SVGP = gpf.models.SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            num_latent_gps=likelihood.latent_dim,
        )

        self.model.compiled_predict_f = tf.function(
            lambda x: self.model.predict_f(x, full_cov=False),
            input_signature=[tf.TensorSpec(shape=[None, 2], dtype=self.dtype)],
        )
        self.model.compiled_predict_y = tf.function(
            lambda x: self.model.predict_y(x, full_cov=False),
            input_signature=[tf.TensorSpec(shape=[None, 2], dtype=self.dtype)],
        )
        self.model.compiled_predict_f_samples = tf.function(
            lambda x, num_samples: self.model.predict_f_samples(x, num_samples, full_cov=False),
            input_signature=[
                tf.TensorSpec(shape=[None, 2], dtype=self.dtype),
                tf.TensorSpec(shape=[None], dtype=np.int64),
            ],
        )

        # Train model:
        dataset = tf.data.Dataset.from_tensor_slices((self.x_samples, self.y_samples))
        batched_dataset = dataset.shuffle(len(dataset), reshuffle_each_iteration=True).batch(batch_size)

        gpf.utilities.set_trainable(self.model.q_mu, False)
        gpf.utilities.set_trainable(self.model.q_sqrt, False)

        variational_vars = [(self.model.q_mu, self.model.q_sqrt)]
        natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

        adam_vars = self.model.trainable_variables
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
            return float(loss_fn())

        log_counter = 0
        k = 5
        last_k_losses = np.zeros(k)
        log_interval = epochs // min(100, epochs)  # log at most 100 times

        for epoch in range(1, epochs + 1):
            avg_batch_loss = 0
            for batch in batched_dataset:
                loss = optimization_step(self.model, batch)
                avg_batch_loss += loss
            avg_batch_loss /= len(batched_dataset)

            last_k_losses[log_counter % k] = avg_batch_loss

            if verbose and epoch % log_interval == 0:
                print(f"Epoch {epoch} - Loss: {avg_batch_loss:.4f}")
                print(f"loc kernel l's: {self.model.kernel.kernels[0].lengthscales}")
                print(f"std kernel l's: {self.model.kernel.kernels[1].lengthscales}")
                log_counter += 1
            if early_stopping and log_counter >= k and early_stop(last_k_losses, verbose=True):
                break

    def _get_whatever(self, x: NDArray[Any], std_factor: float) -> NDArray[Any]:
        if self.model is None:
            raise Exception("Model has not been fitted yet.")
        gp_mean, gp_var = self.model.compiled_predict_y(x)
        gp_mean = gp_mean.numpy()
        gp_std = np.sqrt(gp_var.numpy())
        return gp_mean + std_factor * gp_std

    def get_upper(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return the mean estimate plus 1.96 standard deviations of y at the position(s) x."""
        return self._get_whatever(x, 1.96)

    def get_middle(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return the mean estimate of y at the position(s) x."""
        return self._get_whatever(x, 0)

    def get_lower(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return the mean estimate minus 1.96 standard deviations of y at the position(s) x."""
        return self._get_whatever(x, -1.96)

    def save(self, model_save_path: Path) -> None:
        """Save the model to disk."""
        tf.saved_model.save(self.model, model_save_path)

    def load(self, model_save_path: Path) -> None:
        """Load the model from disk."""
        self.model = tf.saved_model.load(model_save_path)


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
