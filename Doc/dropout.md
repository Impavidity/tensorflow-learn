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


这里过一下tensorflow中dropout的具体实现方式.

```
def dropout(x, keep_prob, noise_shape=None, seed=None, name=None):
  with ops.name_scope(name, "dropout", [x]) as name:  
    x = ops.convert_to_tensor(x, name="x") # 将x转化为tensor
    if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1: # 判断prob是否符合要求
      raise ValueError("keep_prob must be a scalar tensor or a float in the "
                       "range (0, 1], got %g" % keep_prob)
    keep_prob = ops.convert_to_tensor(keep_prob,
                                      dtype=x.dtype,
                                      name="keep_prob") # 将keep_prob转化为tensor, dtype是和x一样的类型
    keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar()) # 确保keep_prob是scalar类型

    # Do nothing if we know keep_prob == 1
    if tensor_util.constant_value(keep_prob) == 1:
      return x

    noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
    # 如果没有生命noise_shape，就让他等于shape(x), 也就是说, 每一维都匹配, 那么每一个元素都与其他无关
    # uniform [keep_prob, 1.0 + keep_prob)
    random_tensor = keep_prob
    random_tensor += random_ops.random_uniform(noise_shape,
                                               seed=seed,
                                               dtype=x.dtype) 
    # 这里有一个十分巧妙的地方. 我们在前面说到, 某一维声明为1的时候, 这一维度一荣俱荣, 一损俱损. 要做到这样, 其实是通过乘法的broadcast.
    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = math_ops.floor(random_tensor)
    ret = math_ops.div(x, keep_prob) * binary_tensor
    ret.set_shape(x.get_shape())
    return ret

```

#### tf.convert_to_tensor(value, dtype=None, name=None, preferred_dtype=None)

将Python里的值转化为tensor. 这个函数接受Tensor, numpy array, python list, 和python scalar.

Example

```
import numpy as np

def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return tf.matmul(arg, arg) + arg

# The following calls are equivalent.
value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
```
这个函数在构造新tensor的时候非常的有用.

- 参数:

  - value: 有相同tensor的值类型的变量.
  - dtype: 新tensor的类型, 可选, 可以从value推理出.
  - name: 新tensor的名字.
  - preferred_dtype: 可选. 新tensor的备选类型(soft preference). 当dtype为None的时候使用. 在一些情况下, caller并不知道新tensor是什么类型. 如果转换到新类型没法完成, 那么这个参数就直接被忽略.

- 返回:

  - 基于value的output.

#### tf.TensorShape.assert_is_compatible_with(other)

当两者不匹配的时候会有Exception出现.











