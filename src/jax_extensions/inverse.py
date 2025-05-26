from functools import wraps

import jax
from jax import lax
from jax.extend import core
from jax.tree_util import tree_flatten, tree_unflatten
from jax._src import pjit
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
  >>>   dst = jnp.flip(src, axis=0)
  >>>   dst = jnp.swapaxes(dst, 0, 1)
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
  def wrapped(*inv_args):
    
    closed_jaxpr = jax.make_jaxpr(fun)(*fwd_args)
    
    # Get the pytree structures for the inverse function outputs
    # i.e. the pytree structures for the forward function inputs
    ret_counts = []
    ret_trees = []
    for fwd_arg in fwd_args:
      fwd_flat_args, ret_tree = tree_flatten(fwd_arg)
      ret_counts.append(len(fwd_flat_args))
      ret_trees.append(ret_tree)

    # Flatten each inverse function input and concatenate them into a single list
    flat_args = []
    arg_trees = []
    for arg in inv_args:
      flat_arg, arg_tree = tree_flatten(arg)
      flat_args.extend(flat_arg)
      arg_trees.append(arg_tree)
    
    # Evaluate the inverse function
    inv_args = flat_args
    out = inverse_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *inv_args)

    # Unflatten the inverse function output
    start_idx = 0
    returns = []
    for count, tree in zip(ret_counts, ret_trees):
      ret = tree_unflatten(tree, out[start_idx:start_idx+count])
      returns.append(ret)
      start_idx += count

    if len(returns) == 1:
      return returns[0]
    return tuple(returns)
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
    outval = inv_fnc(env, eqn, *invals)
    if not isinstance(outval, list):
      outval = [outval]
    safe_map(write, eqn.invars, outval)
    
    # safe_map(write, eqn.invars, [outval])
  return safe_map(read, jaxpr.invars)


def eqn_params(eqn):
  return eqn.primitive.get_bind_params(eqn.params)[1]

def eqn_input_shapes(eqn):
  return [var.aval.shape for var in eqn.invars]

def eqn_input_shape(eqn):
  input_shapes = eqn_input_shapes(eqn)
  assert len(input_shapes) == 1
  return input_shapes[0]

def eqn_invars(eqn):
  return eqn.invars


def inv_rev(env, eqn, val):
  params = eqn_params(eqn)
  return lax.rev(val, params['dimensions'])

def inv_transpose(env, eqn, val):
  params = eqn_params(eqn)
  permutation = params['permutation']
  reversed_permutation = [0] * len(permutation)
  for idx, v in enumerate(permutation):
    reversed_permutation[v] = idx
  return jnp.transpose(val, reversed_permutation)

def inv_reshape(env, eqn, val):
  input_shapes = eqn_input_shape(eqn)
  params = eqn_params(eqn)
  assert params['sharding'] is None
  return jnp.reshape(val, input_shapes)

def inv_pjit(env, eqn, *vals):
  params = eqn_params(eqn)
  closed_jaxpr = params['jaxpr']
  return inverse_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *vals)[0]

def inv_slice(env, eqn, val):
  params = eqn_params(eqn)
  start_indices = params['start_indices']
  limit_indices = params['limit_indices']
  assert params['strides'] is None 
  assert tuple(end - start for end, start in zip(limit_indices, start_indices)) == eqn.outvars[0].aval.shape
  outvar = eqn.invars[0]
  if outvar not in env:
    outval = jnp.zeros(outvar.aval.shape, dtype=outvar.aval.dtype)
    def write(var, val):
      env[var] = val
    safe_map(write, eqn.invars, [outval])
  outval = env[outvar]
  return lax.dynamic_update_slice(outval, val, start_indices)

def inv_squeeze(env, eqn, val):
  params = eqn_params(eqn)
  return jnp.expand_dims(val, params['dimensions'])

def inv_concatenate(env, eqn, val):
  input_shapes = eqn_input_shapes(eqn)
  params = eqn_params(eqn)
  return jnp.split(val, len(input_shapes), axis=params['dimension'])

def inv_broadcast_in_dim(env, eqn, val):
  input_shape = eqn_input_shape(eqn)
  params = eqn_params(eqn)
  broadcast_dimensions = params['broadcast_dimensions']
  shape = params['shape']
  assert params['sharding'] is None

  start_indices = [0] * len(shape)
  limit_indices = [1] * len(shape)
  for idx, dim in enumerate(broadcast_dimensions):
    limit_indices[dim] = input_shape[idx]
  
  squeeze_dimensions = []
  for dim in range(len(shape)):
    if dim not in broadcast_dimensions:
      squeeze_dimensions.append(dim)
  squeeze_dimensions = tuple(squeeze_dimensions)

  return lax.squeeze(
    lax.slice(val, start_indices, limit_indices, None),
    dimensions=squeeze_dimensions
  )

def inv_split(env, eqn, *vals):
  params = eqn_params(eqn)
  return lax.concatenate(vals, dimension=params['axis'])


inverse_registry = {}
inverse_registry[lax.reshape_p] = inv_reshape
inverse_registry[lax.transpose_p] = inv_transpose
inverse_registry[lax.rev_p] = inv_rev
inverse_registry[pjit.pjit_p] = inv_pjit
inverse_registry[lax.slice_p] = inv_slice
inverse_registry[lax.squeeze_p] = inv_squeeze
inverse_registry[lax.concatenate_p] = inv_concatenate
inverse_registry[lax.broadcast_in_dim_p] = inv_broadcast_in_dim
inverse_registry[lax.split_p] = inv_split

