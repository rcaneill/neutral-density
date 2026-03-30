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


def _vectorial_product(A, B):
    """
    A, B: 3D vectors defined at T point as tuple (Ax, Ay, Az) or list or array

    We want this function to be Jax differentiable.
    Note: we do not use this function

    return
    vector defined at T point
    """
    # A[2] *= 1e-5 ## N2
    # B[2] *= 1e-5
    # A and B should have similar order of magnitude in the 3 directions
    # (we look as surfaces that are mostly horizontal)
    out_x = A[2] * B[1] - A[1] * B[2]
    out_y = -A[2] * B[0] + A[0] * B[2]
    out_z = A[1] * B[0] - A[0] * B[1]
    return jax.numpy.array([out_x, out_y, out_z])


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


def loss_at_each_point_v0(
    gamma,
    A,
    e1u,
    e2v,
    e3w,
    weight_per_point,
    mask_gradient,
    normalization=jnp.array([1, 1, 1e-5])[:, jnp.newaxis, jnp.newaxis, jnp.newaxis],
):
    """
    return C at each point
    """
    gradient_gamma = gradients_gamma(gamma, e1u, e2v, e3w) * mask_gradient
    C = jnp.abs(
        jax.numpy.moveaxis(
            cross_product_normalized(
                A,
                gradient_gamma,
                normalization=normalization,
            ),
            -1,
            0,
        )
    )
    return C


def loss1_at_each_point(
    gamma,
    A,
    e1u,
    e2v,
    e3w,
    weight_per_point,
    mask_gradient,
    normalization=jnp.array([1, 1, 1e-5])[:, jnp.newaxis, jnp.newaxis, jnp.newaxis],
):
    """
    return first component of loss at each point
    """
    gradient_gamma = gradients_gamma(gamma, e1u, e2v, e3w) * mask_gradient
    C = jax.numpy.moveaxis(
        cross_product_normalized(
            A,
            gradient_gamma,
            normalization=normalization,
        ),
        -1,
        0,
    )
    return C


def loss2_at_each_point(
    gamma,
    b,
    A,
    e1u,
    e2v,
    e3w,
    weight_per_point,
    mask_gradient,
    normalization=jnp.array([1, 1, 1e-5])[:, jnp.newaxis, jnp.newaxis, jnp.newaxis],
):
    """
    return 2nd component of loss at each point
    """
    gradient_gamma = gradients_gamma(gamma, e1u, e2v, e3w) * mask_gradient
    # C = (b * gradient_gamma - A) * mask_gradient
    C = (gradient_gamma - A) * mask_gradient
    return C


def loss_components(
    gamma,
    b,
    A,
    e1u,
    e2v,
    e3w,
    weight_per_point,
    mask_gradient,
    normalization=jnp.array([1, 1, 1e-5])[:, jnp.newaxis, jnp.newaxis, jnp.newaxis],
):
    """
    return X**2, Y**2, and Z**2
    """
    C1 = (
        loss1_at_each_point(
            gamma, A, e1u, e2v, e3w, weight_per_point, mask_gradient, normalization
        )
        * 0
    )
    C2 = loss2_at_each_point(
        gamma, b, A, e1u, e2v, e3w, weight_per_point, mask_gradient, normalization
    )
    X2 = jnp.nansum((C1[0] ** 2 + C2[0] ** 2) * weight_per_point)
    Y2 = jnp.nansum((C1[1] ** 2 + C2[1] ** 2) * weight_per_point)  # * 1e-6
    Z2 = jnp.nansum((C1[2] ** 2 + C2[2] ** 2) * weight_per_point) * 1e-8
    # works when b=1 is forced, and X2=0
    return jnp.array([X2, Y2, Z2])


def loss(
    gamma,
    b,
    A,
    e1u,
    e2v,
    e3w,
    weight_per_point,
    mask_gradient,
    normalization=jnp.array([1, 1, 1e-5])[:, jnp.newaxis, jnp.newaxis, jnp.newaxis],
):
    return jax.numpy.nansum(
        loss_components(
            gamma, b, A, e1u, e2v, e3w, weight_per_point, mask_gradient, normalization
        )
    )


def tilde_m1(gamma, omega1=0, omega2=1):
    """
    Transform gamma to gamma tilde

    This transformation can be used 1) for the minimisation process,
    and 2) after the minimisation is done to introduce back the physics
    (e.g. having values close to 26 to be consistent with older definitions
    of neutral density).

    Return gamma tilde = (omega2 - omega1)/(max(gamma) - min(gamma)) * (gamma - min(gamma)) + omega1

    Parameters
    ----------
    gamma: jnp.array
    omega1: number
    omega2: number

    Returns
    -------
    jnp.array
    """
    return (omega2 - omega1) / (jnp.max(gamma) - jnp.min(gamma)) * (
        gamma - jnp.min(gamma)
    ) + omega1
