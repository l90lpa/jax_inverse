from functools import wraps

import jax
from jax import lax
from jax._src import pjit
from jax.extend import core
from jax._src.util import safe_map
import jax.numpy as jnp


def inverse(fun, *fwd_args):
  """Compute the inverse of ``fun``.

  Args:
    fun: The function to be inverted. Similar to the VJP transform, its arguments
      should be arrays, scalars, or standard Python containers of arrays or scalars.
      It should return an array, scalar, or standard Python container of arrays or scalars.
    fwd_args: A sequence of forward values as example to determine the
      shapes of intermediate values in ``fun``.

  Returns:
    If the inverse function of ``fun``.

  >>> import jax.numpy as jnp
  >>> from jax_ext_inverse import inverse
  >>> 
  >>> def g(src):
  >>>   dst = jnp.flip(src, axis=0) # flip latitude axis
  >>>   dst = jnp.swapaxes(dst, 0, 1) # (lat, lon)
  >>>   dst = jnp.reshape(dst, (dst.shape[0] * dst.shape[1], 1))
  >>>   return dst
  >>> 
  >>> x = jnp.array([[1, 2, 3], [4, 5, 6]])
  >>> y = g(x)
  >>> g_inv = inverse(g, x)
  >>> z = g_inv(y)
  >>> assert jnp.allclose(x, z)
  """
  @wraps(fun)
  def wrapped(*args):
    # Since we assume unary functions, we won't worry about flattening and
    # unflattening arguments.
    closed_jaxpr = jax.make_jaxpr(fun)(*fwd_args)
    out = inverse_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
    return out[0]
  return wrapped

def inverse_jaxpr(jaxpr, consts, *args):
  env = {}

  def read(var):
    if type(var) is core.Literal:
      return var.val
    return env[var]

  def write(var, val):
    env[var] = val
  # Args now correspond to Jaxpr outvars
  safe_map(write, jaxpr.outvars, args)
  safe_map(write, jaxpr.constvars, consts)

  # Looping backward
  for eqn in reversed(jaxpr.eqns):
    # outvars are now invars
    invals = safe_map(read, eqn.outvars)

    if eqn.primitive in inverse_registry:
      inv_fnc = inverse_registry[eqn.primitive]
    else:
      raise NotImplementedError(
          f"{eqn.primitive} does not have registered inverse.")

    # Assuming a unary function
    outval = inv_fnc(eqn, *invals)
    safe_map(write, eqn.invars, [outval])
  return safe_map(read, jaxpr.invars)


def eqn_params(eqn):
  return eqn.primitive.get_bind_params(eqn.params)[1]

def eqn_input_shapes(eqn):
  return [var.aval.shape for var in eqn.invars]


def inv_rev(eqn, val):
  params = eqn_params(eqn)
  dimensions = params['dimensions']
  return lax.rev(val, dimensions)

def inv_transpose(eqn, val):
  params = eqn_params(eqn)
  permutation = params['permutation']
  reversed_permutation = [0] * len(permutation)
  for idx, v in enumerate(permutation):
    reversed_permutation[v] = idx
  return jnp.transpose(val, reversed_permutation)

def inv_reshape(eqn, val):
  input_shapes = eqn_input_shapes(eqn)
  assert len(input_shapes) == 1
  return jnp.reshape(val, input_shapes[0])

def inv_pjit(eqn, *vals):
  params = eqn_params(eqn)
  closed_jaxpr = params['jaxpr']
  return inverse_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *vals)[0]

inverse_registry = {}
inverse_registry[lax.reshape_p] = inv_reshape
inverse_registry[lax.transpose_p] = inv_transpose
inverse_registry[lax.rev_p] = inv_rev
inverse_registry[pjit.pjit_p] = inv_pjit


