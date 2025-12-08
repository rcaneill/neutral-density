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


def cross_product_normalized(
    A, B, normalization=jnp.array([1, 1, 1])[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
):
    """
    Compute the cross product of A and B, with optional normalization

    Parameters
    ----------
    A, B: jax.numpy.array
        vectors defined at T point, with dimensions in order [x, y, z]
        4d array with dimensions in order (component along X, Y, or Z, location X, location Y, location Z)
    normalization: jax.numpy.array
        Normalization to apply before computing the cross product, optionnal
        Must be broadcastable onto A and B

    Returns
    -------
    4d jax.numpy.array
        product, with dimensions in order (component along X, Y, or Z, location X, location Y, location Z)
    """
    # return jax.numpy.cross(A * [1, 1, 1e-5], B * [1, 1, 1e-5], axisa=0, axisb=0)
    # we don’t simply multiply by the normalization afterward, for numerical precision
    return jax.numpy.cross(A * normalization, B * normalization, axisa=0, axisb=0)


def loss_at_each_point(gamma, A, e1u, e2v, e3w, weight_per_point, mask_gradient):
    """
    return C at each point
    """
    C = (
        jax.numpy.moveaxis(
            cross_product_normalized(
                A,
                gradients_gamma(gamma, e1u, e2v, e3w),
                normalization=jnp.array([1, 1, 1e-5])[
                    :, jnp.newaxis, jnp.newaxis, jnp.newaxis
                ],
            ),
            -1,
            0,
        )
        * mask_gradient
    )
    return C


def loss_components(gamma, A, e1u, e2v, e3w, weight_per_point, mask_gradient):
    """
    return X**2, Y**2, and Z**2
    """
    C = loss_at_each_point(gamma, A, e1u, e2v, e3w, weight_per_point, mask_gradient)
    X2 = jnp.nansum(C[0] ** 2 * weight_per_point)
    Y2 = jnp.nansum(C[1] ** 2 * weight_per_point)
    Z2 = jnp.nansum(C[2] ** 2 * weight_per_point)
    return jnp.array([X2, Y2, Z2])


def loss(gamma, A, e1u, e2v, e3w, weight_per_point, mask_gradient):
    return jax.numpy.nansum(
        loss_components(gamma, A, e1u, e2v, e3w, weight_per_point, mask_gradient)
    )
