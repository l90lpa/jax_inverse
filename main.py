import jax.numpy as jnp
from jax_ext_inverse import inverse

def g(src):
  dst = jnp.flip(src, axis=0) # flip latitude axis
  dst = jnp.swapaxes(dst, 0, 1) # (lat, lon)
  dst = jnp.reshape(dst, (dst.shape[0] * dst.shape[1], 1))
  return dst

x = jnp.array([[1, 2, 3], [4, 5, 6]])
y = g(x)
g_inv = inverse(g, x)
z = g_inv(y)

print(f"Original x:\n{x}")
print(f"Transformed y:\n{y}")
print(f"Inverse z:\n{z}")
print(f"Inverse z equals original x: {jnp.allclose(x, z)}")