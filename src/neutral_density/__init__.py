import numpy as np
import xarray as xr
import cf_xarray
from xgcm import Grid, generate_grid_ds
import gsw_xarray as gsw

import jax
from jax import numpy as jnp


def gradient_centered(array_T, e1u, e2v, e3w):
    """
    Compute the 3 derivatives of array_T, using centered finite differences

    We want this function to be Jax differentiable

    /! Caution: assumes e3w is defined with "right" convention, like e1u and e2v in NEMO

    Parameters
    ----------
    array_T: 3d jax.numpy.array
        Should have dimension order [x, y, z]
    e1u, e2v, e3w: jax.numpy.array
        The scale factors at U,V,W points

    Returns
    -------
    gradient: 4d jax.numpy.array
        [d_dx, d_dy, d_dz]
    """
    # one should compute 1/e1u once and store in new variables
    dg_dx = (
        jnp.roll(array_T, shift=-1, axis=0) - jnp.roll(array_T, shift=1, axis=0)
    ) / (jnp.roll(e1u, shift=1, axis=0) + e1u)
    dg_dy = (
        jnp.roll(array_T, shift=-1, axis=1) - jnp.roll(array_T, shift=1, axis=1)
    ) / (jnp.roll(e2v, shift=1, axis=1) + e2v)
    dg_dz = (
        jnp.roll(array_T, shift=-1, axis=2) - jnp.roll(array_T, shift=1, axis=2)
    ) / (jnp.roll(e3w, shift=1, axis=2) + e3w)
    return jnp.array([dg_dx, dg_dy, dg_dz])


def gradients_gamma(gamma, e1u, e2v, e3w):
    """
    Compute gradient of gamma

    Parameters
    ---------
    gamma: jax.numpy.array
        3d array, with dimensions in order [x, y, z]
    e1u, e2v, e3t: jax.numpy.array
        scale factors at U, V, W points

    Returns
    -------
    jax.numpy.array
        4d array, with [dgamma_dx, dgamma_dy, dgamma_dz]

    """
    return gradient_centered(gamma, e1u, e2v, e3w)


def error_vector_at_each_point(
    gamma,
    A,
    e1u,
    e2v,
    e3w,
    weight_per_point,
    mask_gradient,
    normalization=jnp.array([1, 1, 1])[:, jnp.newaxis, jnp.newaxis, jnp.newaxis],
):
    """
    return the signed error vector at each point
    """
    gradient_gamma = gradients_gamma(gamma, e1u, e2v, e3w) * mask_gradient
    C = (gradient_gamma - A) * mask_gradient * normalization
    return C


def squarred_error_vector(
    gamma,
    A,
    e1u,
    e2v,
    e3w,
    weight_per_point,
    mask_gradient,
    normalization=jnp.array([1, 1, 1])[:, jnp.newaxis, jnp.newaxis, jnp.newaxis],
):
    """
    return X**2, Y**2, and Z**2
    """
    C = error_vector_at_each_point(
        gamma, A, e1u, e2v, e3w, weight_per_point, mask_gradient, normalization
    )
    X2 = jnp.nansum((C[0] ** 2) * weight_per_point)
    Y2 = jnp.nansum((C[1] ** 2) * weight_per_point)
    Z2 = jnp.nansum((C[2] ** 2) * weight_per_point)
    return jnp.array([X2, Y2, Z2])


def loss(
    gamma,
    A,
    e1u,
    e2v,
    e3w,
    weight_per_point,
    mask_gradient,
    normalization=jnp.array([1, 1, 1e-4])[:, jnp.newaxis, jnp.newaxis, jnp.newaxis],
):
    return jax.numpy.nansum(
        squarred_error_vector(
            gamma, A, e1u, e2v, e3w, weight_per_point, mask_gradient, normalization
        )
    )
