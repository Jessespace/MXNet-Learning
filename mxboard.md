# Logging MXNet Data for Visualization in TensorBoard
## Introduction
TensorBoard将数据写到event file中,然后使用tensorboard提取相关数据,并显示出来.
## Installation
```python
pip install mxboard
pip install tensorflow tensorboard
```
## Python API
```python
 mxboard.SummaryWriter.add_graph
 mxboard.SummaryWriter.add_scalar
 mxboard.SummaryWriter.add_histogram
 mxboard.SummaryWriter.add_embedding
 mxboard.SummaryWriter.add_image
 mxboard.SummaryWriter.add_text
 mxboard.SummaryWriter.add_pr_curve
 mxboard.SummaryWriter.add_audio
```
## Histogram
**Step 1.**  数据写入event file
```python
import mxnet as mx
from mxboard import SummaryWriter

with SummaryWriter(logdir='./logs') as sw:
    for i in range(10):
        # create a normal distribution with fixed mean and decreasing std
        data = mx.nd.normal(loc=0, scale=10.0/(i+1), shape=(10, 3, 8, 8))
        sw.add_histogram(tag='norml_dist', values=data, bins=200, global_step=i)

```
在当前文件夹log中写入event file,如:`events.out.tfevents.1541585492.jesse`

**Step 2.**  Tensorboard显示数据
```python
tensorboard --logdir=./logs --host=127.0.0.1 --port=8888
```

在浏览器中打开`http://127.0.0.1:8888`,如果`8888`端口不可用,可以尝试使用其它端口
![](images/mxboard/histogram.png)

## Graph
### Multi-Layer Perceptron
不同于Symbol, Gluon使用`HybridSequential()`而不是常用的`Sequential()`
```python
from mxnet.gluon import nn
from mxboard import SummaryWriter
import mxnet as mx
net=nn.HybridSequential()
with net.name_scope():
    net.add(nn.Dense(128,activation='relu'))
    net.add(nn.Dense(64,activation='relu'))
    net.add(nn.Dense(10))

net.hybridize()
net.initialize()
X=mx.nd.random.uniform(shape=(100,3))
Y=net(X)
with SummaryWriter(logdir='./logs') as sw:
    sw.add_graph(net)
```

![](./images/mxboard/graph_run.png)

### Inception V3
Tensorboard显示**MXNet Gluon Model Zoo**提供的`inceptionv3`网络
```python
from mxnet.gluon.model_zoo.vision import get_model
from mxboard import SummaryWriter
import mxnet as mx

net = get_model('inceptionv3')
net.hybridize()
net.initialize()
net.forward(mx.nd.ones((1, 3, 299, 299)))

with SummaryWriter(logdir='./logs') as sw:
    sw.add_graph(net)
```
![](./images/mxboard/Inception-30.png)

## Scalar
### 单一曲线绘制
![](./images/mxboard/scalar_sin_function_curve.png)
### 多条曲线绘制
![](./images/mxboard/scalar_multi_curves.png)

## Image
```python
import mxnet as mx
import numpy as np
from mxboard import SummaryWriter
from scipy import misc

# get a raccoon face image from scipy
# and convert its layout from HWC to CHW
face = misc.face().transpose((2, 0, 1))
# add a batch axis in the beginning
face = face.reshape((1,) + face.shape)
# replicate the face by 15 times
faces = [face] * 15
# concatenate the faces along the batch axis
faces = np.concatenate(faces, axis=0)

img = mx.nd.array(faces, dtype=faces.dtype)
with SummaryWriter(logdir='./logs') as sw:
    # write batched faces to the event file
    sw.add_image(tag='faces', image=img)
```
![](./images/mxboard/image_faces.png)

## Embedding
Embedding将数据特征压缩到2Ｄ或者3Ｄ中显示，以观察数据分布规律．
![](./images/mxboard/Embedding.png)

## Audio
```python
import mxnet as mx
from mxboard import SummaryWriter

frequency = 44100
# 44100 random samples between -1 and 1
data = mx.random.uniform(low=-1, high=1, shape=(frequency,))
max_abs_val = data.abs().max()
# rescale the data to the range [-1, 1]
data = data / max_abs_val
with SummaryWriter(logdir='./logs') as sw:
    sw.add_audio(tag='uniform_audio', audio=data, global_step=0)
```
![](./images/mxboard/audio.png)

## PR Curve
```python
import mxnet as mx
import numpy as np
from mxboard import SummaryWriter

with SummaryWriter(logdir='./logs') as sw:
    predictions = mx.nd.uniform(low=0, high=1, shape=(100,), dtype=np.float32)
    labels = mx.nd.uniform(low=0, high=2, shape=(100,), dtype=np.float32).astype(np.int32)
    sw.add_pr_curve(tag='pseudo_pr_curve', predictions=predictions, labels=labels, num_thresholds=120)
```
![](./images/mxboard/PR-curve.png)

## References
1. https://github.com/awslabs/mxboard
2. https://github.com/tensorflow/tensorboard
3. https://github.com/awslabs/mxboard/blob/master/demo.md
