from typing import Callable

import jax.numpy as jnp
from jax import Array
from mujoco._specs import MjsBody, MjsJoint, MjsSite


def filter_attrs(target: MjsBody | MjsJoint | MjsSite) -> list[str]:
    """Filter out all attributes of a MuJoCo object that need to be cloned.

    MuJoCo doesn't provide a way to get a list of valid attributes for an object, so we have to
    manually filter out the invalid ones.

    Args:
        target: The MuJoCo object to filter the attributes of.

    Returns:
        A list of valid attributes for the object.
    """
    attrs = list(k for k in target.__dir__() if not k.startswith("_"))
    attrs = list(a for a in attrs if not isinstance(getattr(target, a), Callable))
    invalid = ["id", "alt"]  # Invalid attributes for any object
    for att in invalid:
        if att in attrs:
            attrs.remove(att)
    return attrs


def clone_body(world: MjsBody, body: MjsBody, name: str) -> MjsBody:
    """Clone a MuJoCo body and all of its children and attach them to the world body.

    MuJoCo doesn't provide a way to clone a body, so we have to do it manually.

    Args:
        world: The MuJoCo world body to attach the cloned body to.
        body: The MuJoCo body to clone.
        name: The name of the cloned body.

    Returns:
        The cloned body.
    """
    new_body = world.add_body(name=name)
    new_body.pos = body.pos
    new_body.quat = body.quat
    # Copy over the contents from the body to the new body
    for i, geom in enumerate(body.geoms):
        # Convert geom attributes to a dict and create new geom
        geom_attrs = {att: getattr(geom, att) for att in filter_attrs(geom)}
        geom_attrs["name"] = f"{name}_geom_{i}"
        new_body.add_geom(**geom_attrs)
    for i, joint in enumerate(body.joints):
        # Convert joint attributes to a dict and create new joint
        joint_attrs = {att: getattr(joint, att) for att in filter_attrs(joint)}
        joint_attrs["name"] = f"{name}_joint_{i}"
        new_body.add_joint(**joint_attrs)
    for i, site in enumerate(body.sites):
        # Convert site attributes to a dict and create new site
        site_attrs = {att: getattr(site, att) for att in filter_attrs(site)}
        site_attrs["name"] = f"{name}_site_{i}"
        new_body.add_site(**site_attrs)
    # Add any other necessary child elements (inertial, etc.)
    body_attrs = filter_attrs(body)
    invalid = ["bodies", "geoms", "joints", "sites", "cameras", "lights", "frames", "name"]
    for attr in body_attrs:
        if attr not in invalid:
            setattr(new_body, attr, getattr(body, attr))
    return new_body


def grid_2d(n: int, spacing: float = 1.0, center: Array = jnp.zeros(2)) -> Array:
    """Generate a 2D grid of points."""
    N = int(jnp.ceil(jnp.sqrt(n)))
    points = jnp.linspace(-0.5 * spacing * (N - 1), 0.5 * spacing * (N - 1), N)
    x, y = jnp.meshgrid(points, points)
    grid = jnp.stack((x.flatten(), y.flatten()), axis=-1) + center
    order = jnp.argsort(jnp.linalg.norm(grid, axis=-1))
    return grid[order[:n]]
