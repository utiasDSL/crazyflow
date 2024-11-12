import jax
import jax.numpy as jnp
import mujoco.mjx as mjx

class DomainRandomizer:
    def __init__(self, env_num: int, seed: int):
        """
        Initializes the domain randomizer with model and seed.

        Parameters:
            env_num (int): Number of environments.
            seed (int): Random seed for reproducibility.
        """
        self.rng = jax.random.PRNGKey(seed)
        self.rng = jax.random.split(self.rng, env_num)

    def randomize_friction_from_distribution(self, sys: mjx.Model, distirbution: str, friction_range: tuple):
        """
        Randomizes the friction coefficient of the system from a uniform distribution.

        Parameters:
            sys (mujoco.mjx.Model): The system model.
            distribution (str): The distribution to sample from.
            friction_range (tuple): The range of friction coefficient.

        Returns:
            batched_sys (mujoco.mjx.Model): The batched randomized system.
        """
        assert distirbution == "uniform", "Only uniform distribution is supported."
        
        def _randomize_friction(key: jax.random.PRNGKey):
            """
            Randomizes the friction coefficient of the system from a uniform distribution.

            Parameters:
                key (jax.random.PRNGKey): Random key.
                friction_range (tuple): The range of friction coefficient.

            Returns:
                res_sys (mujoco.mjx.Model): The randomized system
            """
            minval, maxval = friction_range
            friction = jax.random.uniform(key, shape=(1,), minval=minval, maxval=maxval)
            friction = sys.geom_friction.at[:, 0].set(friction)
            res_sys = sys.tree_replace({
                    'geom_friction': friction
                })
            return res_sys

        batched_sys = jax.vmap(_randomize_friction)(self.rng)

        return batched_sys