"""Implicit Quantile Networks for distributional state transition modeling.

Learns P(s'|s,a) via quantile regression, enabling uncertainty-aware planning.
Key reference: Dabney et al. (2018) "Implicit Quantile Networks for Distributional RL"
Extended here for state transition modeling rather than value estimation.
"""

from typing import Callable, Tuple
import chex
import jax
import jax.numpy as jnp
import optax
from flax import nnx, struct
from flax.training.train_state import TrainState
import flashbax as fbx


class QuantileEmbedding(nnx.Module):
    """Cosine embedding for quantile levels τ ∈ [0,1].
    
    Maps scalar τ to a d-dimensional embedding using cosine basis functions:
        φ(τ) = ReLU(W @ [cos(πiτ) for i in 0..n_cos-1] + b)
    
    :param embed_dim: Output embedding dimension.
    :param n_cos: Number of cosine basis functions.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_cos: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.n_cos = n_cos
        self.embed_fc = nnx.Linear(n_cos, embed_dim, rngs=rngs)

    def __call__(self, tau: jax.Array) -> jax.Array:
        """
        :param tau: Quantile levels, shape (...,) or (..., n_quantiles).
        :return: Embeddings, shape (..., embed_dim) or (..., n_quantiles, embed_dim).
        """
        # Ensure tau has at least 1 dimension
        tau = jnp.atleast_1d(tau)
        
        # Cosine basis: cos(π * i * τ) for i = 0, 1, ..., n_cos-1
        i_pi = jnp.pi * jnp.arange(self.n_cos)  # (n_cos,)
        # Expand tau for broadcasting: (..., 1) @ (n_cos,) -> (..., n_cos)
        cos_features = jnp.cos(tau[..., None] * i_pi)  # (..., n_cos)
        
        # Linear projection with ReLU
        embedding = nnx.relu(self.embed_fc(cos_features))  # (..., embed_dim)
        return embedding


class IQNStateNetwork(nnx.Module):
    """Implicit Quantile Network for state transition prediction.
    
    Given (s, a, τ), predicts the τ-quantile of the next state distribution.
    Architecture:
        1. Encode (s, a) -> hidden representation
        2. Embed τ -> quantile embedding
        3. Combine via element-wise product
        4. Decode -> next state prediction
    
    :param state_dim: Dimension of state space.
    :param action_dim: Dimension of action space.
    :param hidden_dim: Hidden layer dimension.
    :param embed_dim: Quantile embedding dimension.
    :param n_cos: Number of cosine basis functions for quantile embedding.
    :param activation: Activation function for hidden layers.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        embed_dim: int = 64,
        n_cos: int = 64,
        activation: Callable = nnx.relu,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation = activation

        # State-action encoder
        self.encoder_fc1 = nnx.Linear(state_dim + action_dim, hidden_dim, rngs=rngs)
        self.encoder_fc2 = nnx.Linear(hidden_dim, embed_dim, rngs=rngs)

        # Quantile embedding
        self.tau_embed = QuantileEmbedding(embed_dim=embed_dim, n_cos=n_cos, rngs=rngs)

        # Decoder (after combining state-action and quantile embeddings)
        self.decoder_fc1 = nnx.Linear(embed_dim, hidden_dim, rngs=rngs)
        self.decoder_fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.decoder_out = nnx.Linear(hidden_dim, state_dim, rngs=rngs)

    def __call__(
        self, state: jax.Array, action: jax.Array, tau: jax.Array
    ) -> jax.Array:
        """Forward pass predicting quantiles of next state.
        
        :param state: Current state, shape (batch, state_dim) or (state_dim,).
        :param action: Action, shape (batch, action_dim) or (action_dim,).
        :param tau: Quantile levels, shape (batch, n_quantiles) or (n_quantiles,).
        :return: Predicted next state quantiles, shape (batch, n_quantiles, state_dim).
        """
        # Handle unbatched inputs
        squeeze_batch = state.ndim == 1
        if squeeze_batch:
            state = state[None, :]
            action = action[None, :]
            tau = tau[None, :]

        batch_size = state.shape[0]
        n_quantiles = tau.shape[-1] if tau.ndim > 1 else 1

        # Encode state-action pair
        sa = jnp.concatenate([state, action], axis=-1)  # (batch, state_dim + action_dim)
        h = self.activation(self.encoder_fc1(sa))  # (batch, hidden_dim)
        sa_embed = self.activation(self.encoder_fc2(h))  # (batch, embed_dim)

        # Embed quantile levels
        tau_embed = self.tau_embed(tau)  # (batch, n_quantiles, embed_dim)

        # Expand sa_embed for broadcasting with quantiles
        sa_embed = sa_embed[:, None, :]  # (batch, 1, embed_dim)

        # Combine via element-wise product (Hadamard)
        combined = sa_embed * tau_embed  # (batch, n_quantiles, embed_dim)

        # Decode to next state prediction
        h = self.activation(self.decoder_fc1(combined))  # (batch, n_quantiles, hidden_dim)
        h = self.activation(self.decoder_fc2(h))  # (batch, n_quantiles, hidden_dim)
        next_state_quantiles = self.decoder_out(h)  # (batch, n_quantiles, state_dim)

        if squeeze_batch:
            next_state_quantiles = next_state_quantiles.squeeze(0)

        return next_state_quantiles

    def predict_mean(
        self, state: jax.Array, action: jax.Array, n_samples: int = 32
    ) -> jax.Array:
        """Predict expected next state by averaging over quantiles.
        
        :param state: Current state.
        :param action: Action.
        :param n_samples: Number of quantile samples to average.
        :return: Mean predicted next state.
        """
        tau = jnp.linspace(0.5 / n_samples, 1 - 0.5 / n_samples, n_samples)
        if state.ndim > 1:
            tau = jnp.broadcast_to(tau, (state.shape[0], n_samples))
        quantiles = self(state, action, tau)
        return quantiles.mean(axis=-2)  # Average over quantile dimension


@struct.dataclass
class Transition:
    """Transition tuple for experience replay."""
    state: jax.Array
    action: jax.Array
    next_state: jax.Array
    reward: jax.Array
    done: jax.Array


@struct.dataclass
class IQNTrainState:
    """Training state for IQN model."""
    model_state: nnx.State
    opt_state: optax.OptState
    step: int


@struct.dataclass
class IQNTransitionModel:
    """Container for trained IQN model and its graph definition."""
    graphdef: nnx.GraphDef
    params: nnx.State


def pinball_loss(
    predictions: jax.Array, targets: jax.Array, tau: jax.Array
) -> jax.Array:
    """Quantile regression (pinball) loss.
    
    ρ_τ(u) = u * (τ - 𝟙_{u<0}) = max(τ*u, (τ-1)*u)
    
    :param predictions: Predicted quantiles, shape (..., n_quantiles, dim).
    :param targets: Target values, shape (..., dim) or (..., 1, dim).
    :param tau: Quantile levels, shape (..., n_quantiles).
    :return: Scalar loss value.
    """
    # Expand targets if needed: (..., dim) -> (..., 1, dim)
    if targets.ndim == predictions.ndim - 1:
        targets = targets[..., None, :]

    # Residuals: predictions - targets
    u = predictions - targets  # (..., n_quantiles, dim)

    # Expand tau for broadcasting with dim: (..., n_quantiles) -> (..., n_quantiles, 1)
    tau_expanded = tau[..., None]

    # Pinball loss: u * (τ - 𝟙_{u<0})
    loss = u * (tau_expanded - (u < 0).astype(jnp.float32))

    # Mean over all dimensions
    return jnp.abs(loss).mean()


def make_iqn_train(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    embed_dim: int = 64,
    n_cos: int = 64,
    n_quantiles: int = 32,
    buffer_size: int = 100_000,
    batch_size: int = 256,
    lr: float = 3e-4,
    num_updates: int = 1000,
    activation: Callable = nnx.relu,
) -> Callable[[chex.PRNGKey, jax.Array], Tuple[IQNTransitionModel, dict]]:
    """Create a training function for IQN state transition model.
    
    :param state_dim: Dimension of state space.
    :param action_dim: Dimension of action space.
    :param hidden_dim: Hidden layer dimension.
    :param embed_dim: Quantile embedding dimension.
    :param n_cos: Number of cosine basis functions.
    :param n_quantiles: Number of quantiles to sample per update.
    :param buffer_size: Replay buffer size.
    :param batch_size: Training batch size.
    :param lr: Learning rate.
    :param num_updates: Number of gradient updates.
    :param activation: Activation function.
    :return: Training function.
    """

    def train(
        key: chex.PRNGKey, transitions: Transition
    ) -> Tuple[IQNTransitionModel, dict]:
        """Train IQN model on collected transitions.
        
        :param key: Random key.
        :param transitions: Transition data (can be from multiple episodes).
        :return: Trained model and training metrics.
        """
        # Initialize network
        key, model_key = jax.random.split(key)
        model = IQNStateNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            n_cos=n_cos,
            activation=activation,
            rngs=nnx.Rngs(model_key),
        )
        graphdef, params = nnx.split(model)

        # Initialize optimizer
        tx = optax.adam(learning_rate=lr)
        opt_state = tx.init(params)

        # Initialize replay buffer
        buffer = fbx.make_flat_buffer(
            max_length=buffer_size,
            min_length=batch_size,
            sample_batch_size=batch_size,
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )

        # Create dummy transition for buffer init
        dummy_transition = Transition(
            state=jnp.zeros(state_dim),
            action=jnp.zeros(action_dim),
            next_state=jnp.zeros(state_dim),
            reward=jnp.zeros(()),
            done=jnp.zeros(()),
        )
        buffer_state = buffer.init(dummy_transition)

        # Add all transitions to buffer
        # Note: transitions should be structured as (n_transitions, ...)
        def add_transition(buffer_state, transition):
            return buffer.add(buffer_state, transition), None

        buffer_state, _ = jax.lax.scan(add_transition, buffer_state, transitions)

        train_state = IQNTrainState(
            model_state=params,
            opt_state=opt_state,
            step=0,
        )

        def update_step(carry, _):
            train_state, key = carry
            key, sample_key, tau_key = jax.random.split(key, 3)

            # Sample batch from buffer
            batch = buffer.sample(buffer_state, sample_key).experience

            # Sample quantile levels
            tau = jax.random.uniform(tau_key, (batch_size, n_quantiles))

            def loss_fn(params):
                model = nnx.merge(graphdef, params)
                # Predict quantiles: (batch, n_quantiles, state_dim)
                pred_quantiles = model(
                    batch.first.state, batch.first.action, tau
                )
                # Target is actual next state: (batch, state_dim)
                target = batch.first.next_state
                return pinball_loss(pred_quantiles, target, tau)

            loss, grads = jax.value_and_grad(loss_fn)(train_state.model_state)
            updates, new_opt_state = tx.update(
                grads, train_state.opt_state, train_state.model_state
            )
            new_params = optax.apply_updates(train_state.model_state, updates)

            new_train_state = IQNTrainState(
                model_state=new_params,
                opt_state=new_opt_state,
                step=train_state.step + 1,
            )
            return (new_train_state, key), {"loss": loss, "step": train_state.step}

        # Run training loop
        key, train_key = jax.random.split(key)
        (final_state, _), metrics = jax.lax.scan(
            update_step, (train_state, train_key), None, length=num_updates
        )

        trained_model = IQNTransitionModel(
            graphdef=graphdef,
            params=final_state.model_state,
        )

        return trained_model, metrics

    return train


def evaluate_iqn_calibration(
    model: IQNTransitionModel,
    test_transitions: Transition,
    n_quantiles: int = 100,
) -> dict:
    """Evaluate calibration of IQN model on held-out transitions.
    
    A well-calibrated model should have:
    - Proportion of true values below predicted τ-quantile ≈ τ
    
    :param model: Trained IQN model.
    :param test_transitions: Test transition data.
    :param n_quantiles: Number of quantile levels to evaluate.
    :return: Calibration metrics.
    """
    network = nnx.merge(model.graphdef, model.params)
    
    # Evaluate at evenly spaced quantiles
    tau_eval = jnp.linspace(0.01, 0.99, n_quantiles)
    n_samples = test_transitions.state.shape[0]
    
    # Broadcast tau to match batch
    tau_batch = jnp.broadcast_to(tau_eval, (n_samples, n_quantiles))
    
    # Get predicted quantiles
    pred_quantiles = network(
        test_transitions.state,
        test_transitions.action,
        tau_batch,
    )  # (n_samples, n_quantiles, state_dim)
    
    # Expand true next states: (n_samples, state_dim) -> (n_samples, 1, state_dim)
    true_next = test_transitions.next_state[:, None, :]
    
    # For each quantile level τ, compute fraction of true values below prediction
    below_quantile = (true_next < pred_quantiles).astype(jnp.float32)
    empirical_coverage = below_quantile.mean(axis=(0, 2))  # (n_quantiles,)
    
    # Calibration error: |empirical_coverage - tau|
    calibration_error = jnp.abs(empirical_coverage - tau_eval)
    
    return {
        "tau": tau_eval,
        "empirical_coverage": empirical_coverage,
        "calibration_error_mean": calibration_error.mean(),
        "calibration_error_max": calibration_error.max(),
    }
