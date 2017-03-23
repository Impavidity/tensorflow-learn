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


