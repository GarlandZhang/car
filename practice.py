import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

def zero_pad(X, pad):
  return np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values=0)

def conv_single_step(a_slice_prev, W, bias):
  mult = np.multiply(a_slice_prev, W) + bias
  return np.sum(mult)

def conv_forward(A_prev, W, bias, hparameters):
  (m, H_prev, W_prev, channel_size_prev) = A_prev.shape
  (filter_size, filter_size, channel_size_prev, channel_size) = W.shape

  stride = hparameters['stride']
  pad = hparameters['pad']

  height = int((H_prev - filter_size + 2 * pad) / stride) + 1
  width = int((W_prev - filter_size + 2 * pad) / stride) + 1

  result = np.zeros((m, height, width, channel_size))

  A_prev_pad = zero_pad(A_prev, pad)

  for i in range(m):
    ex = A_prev_pad[i]
    for h in range(height):
      for w in range(width):
        for c in range(channel_size):
          vertical_start = h * stride
          vertical_end = vertical_start + filter_size
          horizontal_start = w * stride
          horizontal_end = horizontal_start + filter_size
          a_slice_prev = ex[vertical_start:vertical_end, horizontal_start:horizontal_end, :]
          result[i, h, w, c] = conv_single_step(a_slice_prev, W[...,c], bias[...,c])
  
  cache = (A_prev, W, bias, hparameters)
  return result, cache

def pool_forward(A_prev, hparameters, mode = "max"):
  (m, H_prev, W_prev, channel_size_prev) = A_prev.shape

  filter_size = hparameters['f']
  stride = hparameters['stride']

  height = int(1 + (H_prev - filter_size) / stride)
  width = int(1 + (W_prev - filter_size) / stride)
  channel_size = channel_size_prev

  result = np.zeros((m, height, width, channel_size))

  for i in range(m):
    ex = A_prev[i]
    for h in range(height):
      for w in range(width):
        for c in range(channel_size):
          vertical_start = h * stride
          vertical_end = vertical_start + filter_size
          horizontal_start = w * stride
          horizontal_end = horizontal_start + filter_size
          a_slice_prev = ex[vertical_start:vertical_end, horizontal_start:horizontal_end, c]

          if mode == "max":
            result[i, h, w, c] = np.max(a_slice_prev)
          elif mode == "average":
            result[i, h, w, c] = np.mean(a_slice_prev)
  
  cache = (A_prev, hparameters)

  return result, cache

def conv_backward(dZ, cache):
  (A_prev, W, bias, hparameters) = cache
  (m, H_prev, W_prev, C_prev) = A_prev.shape
  (filter_size, filter_size, channel_size_prev, channel_size) = W.shape

  stride = hparameters['stride']
  pad = hparameters['pad']

  (m, height, width, channel_size) = dZ.shape

  dA_prev = np.zeros((m, H_prev, W_prev, C_prev))
  dW = np.zeros((filter_size, filter_size, channel_size_prev, channel_size))
  db = np.zeros((1, 1, 1, channel_size))

  A_prev_pad = zero_pad(A_prev, pad)
  dA_prev_pad = zero_pad(dA_prev, pad)

  for i in range(m):
    ex = A_prev_pad[i]
    da_prev_pad = dA_prev_pad[i]

    for h in range(height):
      for w in range(width):
        for c in range(channel_size):
          vertical_start = h * stride
          vertical_end = vertical_start + filter_size
          horizontal_start = w * stride
          horizontal_end = horizontal_start + filter_size
          a_slice = ex[vertical_start:vertical_end, horizontal_start:horizontal_end, :]
          da_prev_pad[vertical_start:vertical_end, horizontal_start:horizontal_end, :] += W[:,:,:,c] * dZ[i,h,w,c]
          dW[:,:,:,c] += a_slice * dZ[i,h,w,c]
          db[:,:,:,c] += dZ[i, h, w, c]
    
    dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

  return dA_prev, dW, db

def create_mask_from_window(x):
  return (x == np.max(x))

def distribute_value(dz, shape):
  (height, width) = shape
  average = dz / (height * width)
  a = np.ones(shape) * average
  return a

def pool_backward(dA, cache, mode = "max"):
  (A_prev, hparameters) = cache
  stride = hparameters['stride']
  filter_size = hparameters['f']

  m, H_prev, W_prev, channel_size_prev = A_prev.shape
  m, height, width, channel_size = dA.shape

  dA_prev = np.zeros(A_prev.shape)

  for i in range(m):
    a_prev = A_prev[i]
    for h in range(height):
      for w in range(width):
        for c in range(channel_size):
          vertical_start = h
          vertical_end = vertical_start + filter_size
          horizontal_start = w
          horizontal_end = horizontal_start + filter_size

          if mode == "max":
            a_prev_slice = a_prev[vertical_start:vertical_end, horizontal_start:horizontal_end, c]
            mask = create_mask_from_window(a_prev_slice)
            print(mask.shape)
            print(mask)
            print(dA[i,h,w].shape)
            print(dA_prev[i, vertical_start:vertical_end, horizontal_start:horizontal_end, c])
            dA_prev[i, vertical_start:vertical_end, horizontal_start:horizontal_end, c] += np.multiply(mask, dA[i, h, w, c])
          elif mode == "average":
            shape = (filter_size, filter_size)
            dA_prev[i, vertical_start:vertical_end, horizontal_start:horizontal_end, c] += distribute_value(dA[i, h, w, c], shape)
  
  return dA_prev


### CONV TEST
# np.random.seed(1)
# A_prev = np.random.randn(10, 4, 4, 3)
# W = np.random.randn(2, 2, 3, 8)
# b = np.random.randn(1, 1, 1, 8)
# hparameters = {"pad" : 2,
#                "stride": 1}

# Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
# print("Z's mean =", np.mean(Z))
# print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])


# np.random.seed(1)
# dA, dW, db = conv_backward(Z, cache_conv)
# print("dA_mean =", np.mean(dA))
# print("dW_mean =", np.mean(dW))
# print("db_mean =", np.mean(db))
# print(dA.shape)


### POOL TEST
# np.random.seed(1)
# A_prev = np.random.randn(2, 4, 4, 3)
# hparameters = {"stride" : 1, "f": 4}

# A, cache = pool_forward(A_prev, hparameters)
# print("mode = max")
# print("A =", A)
# print()
# A, cache = pool_forward(A_prev, hparameters, mode = "average")
# print("mode = average")
# print("A =", A)

# np.random.seed(1)
# A_prev = np.random.randn(5, 5, 3, 2)
# hparameters = {"stride" : 1, "f": 2}
# A, cache = pool_forward(A_prev, hparameters)
# dA = np.random.randn(5, 4, 2, 2)

# dA_prev = pool_backward(dA, cache, mode = "max")
# print("mode = max")
# print('mean of dA = ', np.mean(dA))
# print('dA_prev[1,1] = ', dA_prev[1,1])  
# print()
# dA_prev = pool_backward(dA, cache, mode = "average")
# print("mode = average")
# print('mean of dA = ', np.mean(dA))
# print('dA_prev[1,1] = ', dA_prev[1,1])