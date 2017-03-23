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


