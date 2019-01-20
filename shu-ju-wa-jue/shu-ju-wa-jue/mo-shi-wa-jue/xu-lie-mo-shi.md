# 序列模式

## 项集数据和序列数据

首先我们看看项集数据和序列数据有什么不同，如下图所示。

![](../../../.gitbook/assets/20181226155839416.bin)

左边的数据集就是项集数据，在Apriori和FP Tree算法中我们也已经看到过了，每个项集数据由若干项组成，这些项没有时间上的先后关系。而右边的序列数据则不一样，它是由若干数据项集组成的序列。比如第一个序列 $$\langle a(abc)(ac)d(cf)\rangle$$ ，它由a,abc,ac,d,cf共5个项集数据组成，并且这些项集有时间上的先后关系。对于多于一个项的项集我们要加上括号，以便和其他的项集分开。同时由于项集内部是不区分先后顺序的，为了方便数据处理，我们一般将序列数据内所有的项集内部按字母顺序排序。

注:序列模式的序列是指项集是有相互顺序的，但项集内部是没有顺序的。

## 子序列与频繁序列

了解了序列数据的概念，我们再来看看什么是子序列。子序列和我们数学上的子集的概念很类似，也就是说，如果某个序列 $$A$$ 所有的项集在序列 $$B$$ 中的项集都可以找到，则AA就是BB的子序列。当然，如果用严格的数学描述，子序列是这样的：

对于序列 $$a_1,a_2,\dots,a_m$$ 和序列 $$b_1,b_2,\dots,b_n$$ ，如果存在数字序列 $$1\leq j_1\leq j_2\leq \dots \leq j_n \leq m$$ ，满足 $$a_1\subseteq b_{j_1},a_2\subseteq b_{j_2},\dots,a_n\subseteq b_{j_n}$$ ，则称 $$A$$ 是 $$B$$ 的子序列。当然反过来说， $$B$$ 是 $$A$$ 的超序列。

而频繁序列则和我们的频繁项集很类似，也就是频繁出现的子序列。比如对于下图，支持度阈值定义为50%，也就是需要出现两次的子序列才是频繁序列。而子序列 $$\langle(ab)c\rangle$$ 是频繁序列，因为它是图中的第一条数据和第三条序列数据的子序列，对应的位置用蓝色标示。

![](../../../.gitbook/assets/20181226155839436.bin)

## GSP

![](../../../.gitbook/assets/timline-jie-tu-20181011153007.png)

## SPADE

![](../../../.gitbook/assets/timline-jie-tu-20181011153047.png)

## PrefixSpan

PrefixSpan算法的全称是Prefix-Projected Pattern Growth，即前缀投影的模式挖掘。里面有前缀和投影两个词。那么我们首先看看什么是PrefixSpan算法中的前缀prefix。

在PrefixSpan算法中的前缀prefix通俗意义讲就是序列数据前面部分的子序列。如果用严格的数学描述，前缀是这样的：对于序列 $$A=a_1,a_2,\dots,a_n$$ 和序列 $$B=b_1,b_2,\dots,b_m$$ ， $$n\leq m$$ 

![](../../../.gitbook/assets/timline-jie-tu-20181011153149.png)

## CloSpan\(针对closed sequential patterns\)

![](../../../.gitbook/assets/timline-jie-tu-20181011153355.png)

## 基于约束的序列模式挖掘

![](../../../.gitbook/assets/timline-jie-tu-20181011153714.png)

## 基于时间约束的序列模式挖掘

![](../../../.gitbook/assets/timline-jie-tu-20181011153812.png)

## 

