### Active Functions

可以使用的激活函数包括:

- 平滑非线性: sigmoid, tanh, elu, softplus, softsign
- 连续非处处可导: relu, relu6, crelu, relu_x
- 随机正则: dropout

#### tf.nn.relu(features, name=None)

计算max(features, 0)

- 参数:

  - features: A tensor, 必须是以下数据类型: float32, float64, int32, int64, uint8, int16, int8, uint16, half
  - name: 可选

- 返回:

  - A tensor, 和features类型一样

#### tf.nn.relu6(features, name=None)

计算min(max(features, 0), 6)

在这篇[paper](http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf)被使用.

- 参数:

  - features: A tensor, 必须是以下数据类型: float32, float64, int32, int64, uint8, int16, int8, uint16, half
  - name: 可选

- 返回:

  - A tensor, 和features类型一样

### Losses

#### tf.nn.l2_loss(t, name=None)

L2 Loss: output = sum(t ** 2)/2

- 参数:

  - t: A tensor, 必须是以下数据类型: float32, float64, int64, int32, uint8, uint16, int16, int8, complex64, complex128, qint8, quint8, qint32, half. 通常是二维, 但可以是任何维度.
  - name: 可选.

- 返回:

  - A tensor, 0-D, 数据类型和t一样.

#### tf.nn.log_poisson_loss(targets, log_input, compute_full_loss=False, name=None)

#### tf.nn.softmax(logits, dim=-1, name=None)

计算softmax.
对于每一个batch i和class j, 我们有
softmax = exp(logits) / reduce_sum(exp(logits), dim)

- 参数:

  - logits: 非空tensor. 必须是一下数据类型: half, float32, float64.
  - dim: 要做softmax的维度, 通常是最后一维, 即-1
  - name: 可选.

- 返回:

  - A tensor. 一样的数据类型和shape

#### tf.nn.log_softmax(logits, dim=-1, name=None)

计算log softmax.
对于每一个batch i和class j, 我们有
logsoftmax = log(softmax) = logits - log(reduce_sum(exp(logits), dim))

- 参数:

  - logits: 非空tensor. 必须是一下数据类型: half, float32, float64.
  - dim: 要做softmax的维度, 通常是最后一维, 即-1
  - name: 可选.

- 返回:

  - A tensor. 一样的数据类型和shape


### Tensor Transformations

#### tf.transpose(a, perm=None, name='transpose')

根据perm来转置a.
对于返回的shape, 在第i维上的值是输入tensor的第perm[i]维的值. 如果perm未给定, 则设置为(n-1, ..., 0)

Example:

```
# 'x' is [[1 2 3]
#         [4 5 6]]
tf.transpose(x) ==> [[1 4]
                     [2 5]
                     [3 6]]

# Equivalently
tf.transpose(x, perm=[1, 0]) ==> [[1 4]
                                  [2 5]
                                  [3 6]]

# 'perm' is more useful for n-dimensional tensors, for n > 2
# 'x' is   [[[1  2  3]
#            [4  5  6]]
#           [[7  8  9]
#            [10 11 12]]]
# Take the transpose of the matrices in dimension-0
tf.transpose(x, perm=[0, 2, 1]) ==> [[[1  4]
                                      [2  5]
                                      [3  6]]

                                     [[7 10]
                                      [8 11]
                                      [9 12]]]
```

- 参数:

  - a: A tensor
  - perm: 一个list, 记录着置换
  - name: 可选

- 返回:

  - a tensor


