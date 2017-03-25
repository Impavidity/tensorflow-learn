### Optimizer

假设我们的optimizer有minimize这一方法. 函数接口如下

```
def minimize(self, loss, name=None)
```

#### tf.trainable_variables()

返回所有trainable=True的变量. 
当传递trainable=True给Variable()的时候，构造函数会自动地将新变量添加到GraphKeys.TRAINABLE_VARIABLES. 这个函数就是用来方便地得到这些变量.

假设我们有一个Optimizer的minimize函数需要完成. 我们需要为每一个变量计算梯度和更新, 在进行梯度计算之前我们应先取得变量并且检查是否有错.

```
    # Error checking
    var_list = tf.trainable_variables()
    for x_tm1 in var_list:
      if not isinstance(x_tm1, tf.Variable):
        raise TypeError("Argument is not a tf.Variable: %s" % x_tm1)
    if not var_list:
      raise ValueError("No variables to optimize")
    if loss.dtype.base_dtype != tf.float32:
      raise ValueError('Loss is not float32')
```

#### Variable._ref()

返回一个reference指向这一变量.

#### tf.gradients(ys, xs, grad_ys=None, name='gradients', colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None)

对于每一个x在xs中, 求其对于sum of ys的偏导数.

- 参数:
  
  - ys: A tensor 或者 a list of tensor
  - xs: A tensor 或者 a list of tensor
  - grad_ys: 可选, 存着ys的导数, 和ys有一样的长度, 就是ys关于上一级的偏导数. 当grad_ys为None的时候, 它会为每一个在ys中的元素填充上一个由1组成的tensor, 这个tensor的shape和元素的shape相同. 用户可以初始化这个参数. 例如如果用户想为每一个ys里的元素提供不同的weight的话.
  - name: 可选, 用于将所有偏导聚集在一起, 默认为'gradients'
  - colocate_gradients_with_ops: 如果为True, 则colocate 相对应的op的gradients //不知道什么意思
  - gate_gradients: 如果为True, 在return的gradients外加一层tuple.
  - aggregation_method: 声明合并gradients的方法.

- 返回:
  
  - 一个sum(dy/dx)的list, 对于每一个x在xs中.

我们继续minimize的方法, 使用了这个函数之后, 我们就可以得到偏导了.

```
    # Compute gradients
    var_refs = [x_tm1._ref() for x_tm1 in var_list]
    grads = tf.gradients(loss, var_refs,
                                colocate_gradients_with_ops=True,
                                gate_gradients=True,
                                aggregation_method=2)
    for x_tm1, g_t in zip(var_list, grads):
      if g_t is not None:
        if x_tm1.dtype.base_dtype != tf.float32:
          raise ValueError('%s is not float32' % x_tm1.name)
```


#### AggregationMethod

定义了合并gradients的几种方法.

- ADD_N = 0 : 在合并之前, 所有的gradients必须算好.
- DEFAULT = ADD_N
- EXPERIMENTAL_TREE = 1 :
- EXPERIMENTAL_ACCUMULATE_N = 2 ： 以任何顺序合并, 早合并, 早释放, 节约资源.

#### tf.Variable.initialized_value()

返回变量初始化的值.
这个方法应该被使用, 当你想用一个变量初始化另一个变量的值的时候.

- 返回:

  - A tensor, 里面的值是初始化后的变量的值.

Example

```
# Initialize 'v' with a random tensor.
v = tf.Variable(tf.truncated_normal([10, 40]))
# Use `initialized_value` to guarantee that `v` has been
# initialized before its value is used to initialize `w`.
# The random values are picked only once.
w = tf.Variable(v.initialized_value() * 2.0)
```


