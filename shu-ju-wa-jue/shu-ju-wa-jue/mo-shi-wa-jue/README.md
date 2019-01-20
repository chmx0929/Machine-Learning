# 模式挖掘

**模式\(pattern\)：**在一个数据集中经常一起出现的子集、子序列、子结构等

**项集\(Itemset\)：**一个或多个数据组成的集合；k项集\(k-itemset\)：有k项的项集 $$X=\{x_1,\dots,x_k\}$$ 

**\(绝对\)支持度\(count\)：** $$X$$ 在数据集中出现的次数或频率

**\(相对\)支持度\(Support, s\)：** $$X$$ 出现在某数据子集的概率； $$s(X) \geq \sigma_{threshold}$$表示 $$X$$ 为频繁项集

**置信度\(Confidence, c\)：**一个交易含有 $$ X$$ 又有 $$Y$$ 的条件概率 $$c = \frac{sup(X\cup Y)}{sup(X)}$$   

**闭合模式\(Closed Pattern\)：** $$X$$ 为频繁项集且不存在 $$Y \supset X$$和 $$X$$ 有相同支持度

**最大模式\(Max-Pattern\)：** $$X$$ 为频繁项集且不存在 $$Y \supset X$$ 



