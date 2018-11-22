# 社区检测与搜索

|  | Community Detection | Community Search |
| :---: | :---: | :---: |
| Goal | Find all communities with a global criterion |  Find communities for particular persons |
| Cost | Expensive | Less expensive |
| Status | Graphs evolve | Online and dynamic |

## [高频模式子图\(Frequent Pattern Graph\)](https://chmx0929.gitbook.io/machine-learning/shu-ju-wa-jue/shu-ju-wa-jue/mo-shi-wa-jue)

详见数据挖掘-模式挖掘-图模式挖掘部分

## [动态异构网络中的社区演化](http://keg.cs.tsinghua.edu.cn/jietang/publications/TKDE13-Sun-etl-al-co-evolution-of-multi-typed-objects-in-dynamic-networks.pdf)

![Community Evolution Discovery](../../../.gitbook/assets/timline-jie-tu-20181015114928.png)

## [K-Truss](https://arxiv.org/pdf/1205.6693.pdf)

K-truss of graph $$G$$ : the largest subgraph $$H$$  s.t. every edge in $$H$$ is contained in at least $$(k-2)$$ triangles within $$H$$ 

![](../../../.gitbook/assets/timline-jie-tu-20180921165034.png)

### [K-Truss Community Model\(找Community\)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.722.9193&rep=rep1&type=pdf)

\(1\) K-truss: each edge within at least $$(k-2)$$ triangles

\(2\) Edge Connectivity: common edges shared by triangles

\(3\) Maximal Subgraph

![](../../../.gitbook/assets/timline-jie-tu-20180921170938.png)

## [K-core](https://arxiv.org/pdf/cs/0504107.pdf)

A k-core $$H$$ is an induced subgraph of $$G$$ where every node in $$H$$ has at least $$k$$ degree in $$H$$

k-core即指一个图中至少有 $$k$$ 条边的点所有子图集合 （可以是非相连的子图）

The maximal k-core $$H$$ is a k-core that no super graph of $$H$$ is a k-core \(The maximal k-core is unique but can be disconnected\)

最大k-core即一个图中边数最多的点的子图 （可以是非相连的，即 $$v_i$$ 与 $$v_j$$ 都有五条边，全图最多边数，$$v_i$$ 与 $$v_j$$ 可以不相连）

### K-Influential Community\(影响力问题\)

Let $$f(H)$$be the influence of a subgraph $$H$$. A K-Influential Community is an induced subgraph $$H$$ of $$G$$ that meets all the following constraints:

\(1\) Connectivity: $$H$$ is connected

\(2\) Cohesiveness: each node in $$H$$ has degree $$\geq k$$ 

\(3\) Maximal structure: there does not exist $$H'(\supseteq H)$$ such that $$f(H) = f(H')$$ 

### K-Core Persistent Community\(随时间变化Community问题\)

Goal: A network changes over time, try to find k-core communities that are persistent most of the time in a temporal network

#### k-Persistent-Core

A subgraph $$G$$ is considered as k-persistent-core if $$G = \cup G_i$$ for every $$G_i$$ appearing in an interval of $$[t_s,t_e]$$ is a connected k-core

## Source

[https://arxiv.org/pdf/1205.6693.pdf](https://arxiv.org/pdf/1205.6693.pdf)

[https://arxiv.org/pdf/cs/0504107.pdf](https://arxiv.org/pdf/cs/0504107.pdf)

[http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.722.9193&rep=rep1&type=pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.722.9193&rep=rep1&type=pdf)

[http://keg.cs.tsinghua.edu.cn/jietang/publications/TKDE13-Sun-etl-al-co-evolution-of-multi-typed-objects-in-dynamic-networks.pdf](http://keg.cs.tsinghua.edu.cn/jietang/publications/TKDE13-Sun-etl-al-co-evolution-of-multi-typed-objects-in-dynamic-networks.pdf)





