# SPP-net

SPP-net是何恺明于14年撰写的论文，主要是把经典的空间金字塔池化层（Spatial Pyramid Pooling，SPP）结构引入CNN中，从而使CNN可以处理任意size和scale的图片。这种方法不仅提升了分类的准确率，而且还非常适合目标检测，比经典的RNN快速准确。

之所以引进SPP层，主要原因是CNN的全连接层要求输入图片的大小一致，而实际中的输入图片往往大小不一，如果直接缩放到同一尺寸，很可能有的物体会充满满张图片，而有的物体只占到图片的一角。

![](../../../../.gitbook/assets/20150202210128332.bin)

## Source

{% embed url="https://blog.csdn.net/whiteinblue/article/details/43415035" %}





