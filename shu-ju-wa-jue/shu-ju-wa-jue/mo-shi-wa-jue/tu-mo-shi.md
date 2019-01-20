# 图模式

## 方法分类

候选集生成方式：Apriori vs. Pattern growth \(FSG vs. gSpan\)

搜索顺序：广度 vs. 深度

重复子图剔除：被动 vs. 主动\(gSpan\)

支持度计算：GASTON, FFSM, MoFa

模式发现顺序：Path-&gt;Tree-&gt;Graph \(GASTON\)

## 基于Apriori的方法

候选集生成 -&gt; 候选集剪枝 -&gt; 支持度计算 -&gt; 候选集剔除  迭代这四步至无法生成候选集或不满足支持度

候选集生成时扩展节点\(AGM算法\)还是扩展边\(FSG算法\)都可以，但是经测试是扩展边更高效

## 基于Pattern-Growth的方法

按深度优先来扩展边，从k边子图-&gt;\(k+1\)边子图-&gt;\(k+2\)边子图...

问题：这样会生成很多重复子图

解决：1、定义一个子图生成顺序  2、DFS生成树，用深度优先搜索扁平图  3、gSpan

#### gSpan

![](../../../.gitbook/assets/timline-jie-tu-20181011160446.png)

## 闭合图模式挖掘

如果不存在与高频图 $$G$$ 有相同支持度的父图 $$G'$$ ，则 $$G$$ 是闭合的；算法：CloseGraph

![](../../../.gitbook/assets/timline-jie-tu-20181011160823.png)

## 

