### GPU

#### 允许GPU Memory增长

有两种方式:

```
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
```

这一种方式在刚开始分配比较小的GPU Memory, 随后随着需求的增加, 它会慢慢增加GPU Memory的占用. 注意, 它不会释放GPU Memory.

```
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config, ...)
```

第二种方式就是预设GPU Memory占用量. 这种方式很麻烦, 因为总是不知道自己程序需要多少GPU.


