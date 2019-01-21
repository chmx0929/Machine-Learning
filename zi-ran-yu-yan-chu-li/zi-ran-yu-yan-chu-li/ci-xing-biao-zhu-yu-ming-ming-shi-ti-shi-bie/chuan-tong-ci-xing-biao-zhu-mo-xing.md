# 传统词性标注模型

传统的词性标注方法有隐马尔可夫模型（HMM）和最大熵马尔可夫模型（MEMM）等。其中，HMM是生成模型，MEMM是判别模型。

基于MEMM的词性标注器抽取当前待标注单词附近的特征，然后利用这些特征判别当前词的特性。MEMM是最大熵模型（ME）在处理序列模型方面的变种。其思想是在一串满足约束的标签中选出一个熵最大的标签。用 $$T$$ 表示标签集合， $$t$$ 表示其中的某一个标签， $$h$$ 表示当前单词的上下文信息。这种模型可以用来估计句子 $$w_1,\dots,w_n$$ 的标注 $$t_1,\dots,t_n$$ 的概率，公式如下：

                          $$p(t_1\dots t_n|w_1\dots w_n)=\prod\limits_{i=1}^np(t_i|t_1\dots t_{i-1},w_1\dots w_n)\approx \prod\limits_{i=1}^np(t_i|h_i)$$ 

当前单词的上下文信息又叫作特征。根据在语料中出现的频次，可以将单词分为常见词和罕见词。常见词周围的特包括：待标注的单词、待标注单词附近的单词、待标注单词附近已标注单词的词性标签等；罕见词的特征包括：单词的后缀、单词的前缀、单词是否包含数字、单词是否首字母大写等。

模型可以使用最大似然的方法来训练，公式如下：

       $$\hat{T}=\mathop{\arg \max}\limits_T P(T|W)=\mathop{\arg \max}\limits_T P(t_1\dots t_n|w_1\dots w_n)\approx \mathop{\arg \max}\limits_T\prod_{i=1}^nP(t_i|h_i)$$ 

            $$ = \mathop{\arg \max}\limits_T\prod_{i=1}^n\frac{\prod_{j=1\dots K}\exp(w_jf_j(h_i,t_i))}{\sum_{t'_i\in T}\prod_{j=1\dots K}\exp(w_jf_j(h_i,t'_i))}$$ 

HMM模型与MEMM模型的概率表示和求解都不相同。HMM前面机器学习-概率图模型章节已经介绍很清楚，在此不再赘述。基于HMM的词性标注模型的目标函数如下：

  $$\hat{T}=\mathop{\arg \max}\limits_T P(T|W)=\mathop{\arg \max}\limits_T P(t_1\dots t_n|w_1\dots w_n)\approx \mathop{\arg \max}\limits_T \prod\limits_{i=1}^n  \overbrace{P(w_i|t_i)}^{发射概率}\overbrace{P(t_i|t_{t-1})}^{转移概率}$$ 

HMM和MEMM存在同一个问题，就是只能从一个方向预测接下来的标注。但是在很多情况下当前单词后面的标注信息对当前标注的反馈也非常重要。要解决这个问题有很多种方法。一种方法是采用更强的模型，比如条件随机场（CRF）。但是条件随机场的计算开销太大，并且对标注效果的提升有限。还有一种方法是斯坦福的词性标注器中用到的，一种叫作循环依赖网络（Cyclic Dependency Network）的模型，这种模型本质上式MEMM的变种。

## Source

{% embed url="https://arxiv.org/abs/1103.0398" %}



