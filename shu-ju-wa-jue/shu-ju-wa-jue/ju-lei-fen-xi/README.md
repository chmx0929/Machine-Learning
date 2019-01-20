# 聚类分析

## 性能度量

聚类是将样本集 $$D$$ 划分为若干不想交的子集，即样本簇。那么，什么样的聚类结果比较好呢？直观上看，我们希望“物以类聚”，即同一簇的样本尽可能彼此相似，不同簇的样本尽可能不同。换言之，聚类结果的“簇内相似度”高且“簇间相似度”低。

聚类性能度量大致有两大类，一类是将聚类结果与某个“参考模型”进行比较，称为“外部指标”；另一类是直接考察聚类结果而不利用任何参考模型，称为“内部指标”。

### 外部指标

Jaccard系数\(Jaccard Coefficient, JC\)： $$JC=\frac{a}{a+b+c}$$ 

FM指数\(Fowlkes and Mallows Index, FMI\)： $$FMI = \sqrt{\frac{a}{a+b}\cdot \frac{a}{a+c}}$$ 

Rand指数\(Rand Index, RI\)： $$RI=\frac{2(a+d)}{m(m-1)}$$ 

显然，上述性能度量的结果值均在 $$[0,1]$$ 区间，值越大越好

### 内部指标

DB指数\(Davies-Bouldin Index, DBI\)： $$DBI=\frac{1}{k}\sum\limits_{i=1}^k(\frac{avg(C_i)+avg(C_j)}{f_{cen}(\mu_i,\mu_j)})$$ 

Dunn指数\(Dunn Index, DI\)： $$DI=\min\limits_{1\leq i\leq k}\{\min\limits_{j\neq i}(\frac{d_{\min}(C_i,C_j)}{\max_{1\leq l\leq k}diam(C_l)})\}$$ 

显然，DBI的值越小越好，而DI则相反，值越大越好



