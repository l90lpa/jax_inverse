from functools import partial

import jax.numpy as jnp
from jax_ext_inverse import inverse

def test_array():

  def f(src):
    dst = jnp.flip(src, axis=0)
    dst = jnp.swapaxes(dst, 0, 1)
    dst = jnp.reshape(dst, (dst.shape[0] * dst.shape[1], 1))
    return dst

  x = jnp.array([[1, 2, 3], [4, 5, 6]])
  y = f(x)

  f_inv = inverse(f, x)
  
  z = f_inv(y)

  # print(f"Original x:\n{x}")
  # print(f"Transformed y:\n{y}")
  # print(f"Inverse z:\n{z}")

  assert jnp.allclose(x, z)

def test_dict():

  ctx = {"x1": ["y1", "y2"], "x2": ["y3"]}
  def f_ctx(ctx, src):
    dst = {}
    for k, v in src.items():
      new_ks = ctx[k]
      for idx, new_k in enumerate(new_ks):
        dst[new_k] = v[idx]
    return dst
  f = partial(f_ctx, ctx)

  x = {"x1": jnp.array([[1,2,3],[4,5,6]]), "x2": jnp.array([[7,8,9],])}
  y = f(x)

  f_inv = inverse(f, x)

  z = f_inv(y)

  # print(f"Original x:\n{x}")
  # print(f"Transformed y:\n{y}")
  # print(f"Inverse z:\n{z}")

  assert x.keys() == z.keys()
  for k in x.keys():
    assert jnp.allclose(x[k], z[k])


def test_multiple_args():

  def f(src1, src2, src3):
    return jnp.stack([src1, src2, src3], axis=1)

  x1 = jnp.array([[1, 2, 3], [4, 5, 6]])
  x2 = jnp.array([[7, 8, 9], [10, 11, 12]])
  x3 = jnp.array([[13, 14, 15], [16, 17, 18]])
  y = f(x1, x2, x3)
  
  f_inv = inverse(f, x1, x2, x3)
  
  z1, z2, z3 = f_inv(y)
  
  # print(f"Original x1:\n{x1}")
  # print(f"Original x2:\n{x2}")
  # print(f"Transformed y:\n{y}")
  # print(f"Inverse z1:\n{z1}")
  # print(f"Inverse z2:\n{z2}")

  assert jnp.allclose(x1, z1)
  assert jnp.allclose(x2, z2)
  assert jnp.allclose(x3, z3)


def test_multiple_rets():

  def f(x):
    return jnp.split(x, 2, axis=0)

  x = jnp.array([[1, 2], [3, 4]])
  y1, y2 = f(x)
  
  f_inv = inverse(f, x)
  
  z = f_inv(y1, y2)
  
  # print(f"Original x:\n{x}")
  # print(f"Transformed y1:\n{y1}")
  # print(f"Transformed y2:\n{y2}")
  # print(f"Inverse z:\n{z}")

  assert jnp.allclose(x, z)


if __name__ == "__main__":
  test_array()
  test_dict()
  test_multiple_args()
  test_multiple_rets()