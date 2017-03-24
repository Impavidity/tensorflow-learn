#### tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)

在keep_prob的概率下, 输入的每一个元素会变成原来的1/keep_prob倍(为了保证输出的平均值或和不变), 否则就设置为0.
默认下, 每一个元素keep or drop是独立的. 如果noise_shape被声明, 它必须是broadcastable到x的shape的.
在此情况下, 只有noise_shape[i] == shape(x)[i], 这个维度才会作出独立判断. 
例如shape(x) = [k, l, m, n], noise_shape = [k, 1, 1, n], 那么每一个batch和每一个channel都会独立判断, 而每一行和每一列会一起判断.

- 参数:
  
  - x: A tensor
  - keep_prob: A scalar Tensor. 和x是相同数据类型.
  - noise_shape: 1-D Tensor. 数据类型是int32.
  - seed: Python integer. 用来设置random seeds.
  - name: 可选

- 返回:

  - 和x一样shape的tensor.



