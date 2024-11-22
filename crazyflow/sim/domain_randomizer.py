import jax
import mujoco.mjx as mjx


class DomainRandomizer:
    def __init__(self, env_num: int, seed: int):
        """Initialize the domain randomizer with seed.

        Parameters:
            env_num (int): Number of environments.
            seed (int): Random seed for reproducibility.
        """
        self.rng = jax.random.PRNGKey(seed)
        self.rng = jax.random.split(self.rng, env_num)
    
    def randomize_friction_from_distribution(
        self, sys: mjx.Model, distirbution: str, friction_range: tuple
    ) -> mjx.Model:
        """Randomize the friction coefficient of the system from a uniform distribution.

        Parameters:
            sys: The system model.
            distribution: The distribution to sample from.
            friction_range: The range of friction coefficient.

        Returns:
            batched_sys: The batched randomized system.
        """
        assert distirbution == "uniform", "Only uniform distribution is supported."

        def _randomize_friction(key: jax.random.PRNGKey) -> mjx.Model:
            """Randomize the friction coefficient of the system from a uniform distribution.

            Parameters:
                key: Random key.

            Returns:
                res_sys: The randomized system
            """
            minval, maxval = friction_range
            friction = jax.random.uniform(key, shape=(1,), minval=minval, maxval=maxval)
            friction = sys.geom_friction.at[:, 0].set(friction)
            res_sys = sys.tree_replace({"geom_friction": friction})
            return res_sys

        batched_sys = jax.vmap(_randomize_friction)(self.rng)

        return batched_sys
    
    def randomize_friction_from_values(
        self, sys: mjx.Model, friction_values: jax.Array
    ) -> mjx.Model:
        """Randomize the friction coefficient of the system from a list of values.

        Parameters:
            sys: The system model.
            friction_values: The list of friction coefficient values.

        Returns:
            batched_sys: The batched randomized system.
        """
        def _randomize_friction(friction: float) -> mjx.Model:
            """Randomize the friction coefficient of the system from a list of values.

            Parameters:
                friction: Friction coefficient value.

            Returns:
                res_sys: The randomized system
            """
            friction = sys.geom_friction.at[:, 0].set(friction)
            res_sys = sys.tree_replace({"geom_friction": friction})
            return res_sys
        
        batched_sys = jax.vmap(_randomize_friction)(friction_values)

        return batched_sys
    
    def randomize_mass_from_distribution(
        self, sys: mjx.Model, distirbution: str, mass_range: tuple
    ) -> mjx.Model:
        """Randomize the mass of the system from a uniform distribution.

        Parameters:
            sys: The system model.
            distribution: The distribution to sample from.
            mass_range: The range of mass.

        Returns:
            batched_sys: The batched randomized system.
        """
        assert distirbution == "uniform", "Only uniform distribution is supported."

        def _randomize_mass(key: jax.random.PRNGKey) -> mjx.Model:
            """Randomize the mass of the system from a uniform distribution.

            Parameters:
                key: Random key.

            Returns:
                res_sys: The randomized system
            """
            minval, maxval = mass_range
            mass = jax.random.uniform(key, shape=(), minval=minval, maxval=maxval)
            mass = sys.body_mass.at[1].set(mass)
            res_sys = sys.tree_replace({"body_mass": mass})
            return res_sys

        batched_sys = jax.vmap(_randomize_mass)(self.rng)

        return batched_sys
    
    def randomize_mass_from_values(
        self, sys: mjx.Model, mass_values: jax.Array
    ) -> mjx.Model:
        """Randomize the mass of the system from a list of values.

        Parameters:
            sys: The system model.
            mass_values: The list of mass values.

        Returns:
            batched_sys: The batched randomized system.
        """
        def _randomize_mass(mass: float) -> mjx.Model:
            """Randomize the mass of the system from a list of values.

            Parameters:
                mass: Mass value.

            Returns:
                res_sys: The randomized system
            """
            mass = sys.body_mass.at[1].set(mass)
            res_sys = sys.tree_replace({"body_mass": mass})
            return res_sys
        
        batched_sys = jax.vmap(_randomize_mass)(mass_values)

        return batched_sys
    
    def randomize_body_positions_from_distribution(
        self, sys: mjx.Model, distirbution: str, perturb_range: tuple
    ) -> mjx.Model:
        """Randomize positions of bodies in the system from a uniform distribution.

        Parameters:
            sys: The system model.
            distribution: The distribution to sample from.
            mass_range: The range of mass.

        Returns:
            batched_sys: The batched randomized system.
        """
        assert distirbution == "uniform", "Only uniform distribution is supported."

        def _perturb_positions(key: jax.random.PRNGKey) -> mjx.Model:
            """Perturb the positions of the bodies in the system.

            Parameters:
                key: Random key.

            Returns:
                res_sys: The randomized system.
            """
            min_offset, max_offset = perturb_range
            perturbations = jax.random.uniform(key, shape=sys.body_pos.shape, minval=min_offset, maxval=max_offset)
            new_body_pos = sys.body_pos + perturbations
            res_sys = sys.tree_replace({"body_pos": new_body_pos})
            return res_sys

        # Apply perturbation across batched keys
        batched_sys = jax.vmap(_perturb_positions)(self.rng)

        return batched_sys
    
    def randomize_body_positions_from_values(
        self, sys: mjx.Model, perturb_values: jax.Array
    ) -> mjx.Model:
        """Randomize positions of bodies in the system from a list of values.

        Parameters:
            sys: The system model.
            perturb_values: The list of perturbation values.

        Returns:
            batched_sys: The batched randomized system.
        """
        def _perturb_positions(perturb: jax.Array) -> mjx.Model:
            """Perturb the positions of the bodies in the system.

            Parameters:
                perturb: Perturbation values.

            Returns:
                res_sys: The randomized system.
            """
            new_body_pos = sys.body_pos + perturb
            res_sys = sys.tree_replace({"body_pos": new_body_pos})
            return res_sys

        # Apply perturbation across batched values
        batched_sys = jax.vmap(_perturb_positions)(perturb_values)

        return batched_sys