#### tf.nn.embedding_lookup(params, ids, partition_strategy='mod', name=None, validate_indices=True, max_norm=None)

这是一个非常基础的函数, 作用在embedding tensors里面找与ids想对应的向量.

#### tf.pack(values, axis=0, name='pack')

将rank-R的tensor堆叠成rank-(R+1)的tensor.
假设我们有长度为N的tensor list, 每一个tensor有shape=(A, B, C).
如果axis==0, 则输出的tensor有shape=(N, A, B, C). 如果axis==1, 则输出的tensor有shape=(A, N, B, C)
Example:

```
# 'x' is [1, 4]
# 'y' is [2, 5]
# 'z' is [3, 6]
pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
```

参数:

- values: Tensor List, 有相同的shape和type
- axis: 堆叠的坐标轴
- name: 可选

返回:

- output: 堆叠后的tensor, 与values有相同的type.

#### Broadcasting

这里给一些broadcasting的例子, 有助于理解其含义.

```
A      (4d array):  8 x 1 x 6 x 1
B      (3d array):      7 x 1 x 5
Result (4d array):  8 x 7 x 6 x 5

A      (2d array):  5 x 4
B      (1d array):      1
Result (2d array):  5 x 4

A      (2d array):  5 x 4
B      (1d array):      4
Result (2d array):  5 x 4

A      (3d array):  15 x 3 x 5
B      (3d array):  15 x 1 x 5
Result (3d array):  15 x 3 x 5

A      (3d array):  15 x 3 x 5
B      (2d array):       3 x 5
Result (3d array):  15 x 3 x 5

A      (3d array):  15 x 3 x 5
B      (2d array):       3 x 1
Result (3d array):  15 x 3 x 5
```

#### tf.placeholder(dtype, shape=None, name=None)

先对于Variable, 这是一个放Variable的容器. 你可以放进这个容器(tensor)随意的值 只要符合type和shape.

它的值必须通过feed_dict来传递给Session.run(), Tensor.eval(), 或者Operation.run().

Example

```
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
  print(sess.run(y))  # ERROR: will fail because x was not fed.

  rand_array = np.random.rand(1024, 1024)
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
```

- 参数:
  
  - dtype: 变量类型
  - shape: 这一选项是可选的, 如果没有被声明, 任何shape任意形状的tensor
  - name: 可选

- 返回:

  - 返回一个Tensor, 用来解决值的feeding问题. 不能直接可数值化(evaluated directly).


### Initializer

#### tf.orthogonal_initializer

使用SVD生成一个正交矩阵来初始化. [[See why here]](blog.impavidity.com/tech/whysInNeuralNetwork/orthogonalInitialization.html)

如果被初始化的tensor是2-D, 那么直接初始化. 如果被初始化的tensor是2维以上的, 那么shape为(shape[0]\*...\*shape[n-2], shape[-1])的矩阵会被初始化. 之后这个矩阵会被reshape回原来的shape.

- 方法: 

  - __init__(gain=1.0, dtype=tf.float32, seed=None)

- 参数:
  
  - gain: 想要应用到正交矩阵的乘子. 默认为1.
  - dtype: 数据类型.
  - seed: Python integer. 用来创建random seed.

#### tf.random_uniform_initializer

用Uniform distribution来初始化.

- 方法:
 
  - __init__(minval=0, maxval=None, seed=None, dtype=tf.float32)

- 参数:

  - minval: A python integer 或者 a scalar tensor. 分布的下限.
  - maxval: A python integer 或者 a scalar tensor. 分布的上限.
  - seed: A python integer. 用来创建random seed.
  - dtype: 数据类型.

#### tf.zeros_initializer

初始化为0.

- 方法:
  
  - __init__(dtype=tf.float32)

#### tf.ones_initializer

初始化为1.

- 方法:

  - __init__(dtype=tf.float32)

#### tf.uniform_unit_scaling_initializer

#### tf.random_normal_initializer

用正态分布初始化.

- 方法:

  - __init__(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)

- 参数:

  - mean: A python integer 或者 a scalar tensor. 均值.
  - stddev:  python integer 或者 a scalar tensor. 标准差.
  - seed: A python integer. 用来创建random seed.
  - dtype: 数据类型. 只有float被支持.

#### tf.constant_initializer

#### tf.truncated_normal_initializer



