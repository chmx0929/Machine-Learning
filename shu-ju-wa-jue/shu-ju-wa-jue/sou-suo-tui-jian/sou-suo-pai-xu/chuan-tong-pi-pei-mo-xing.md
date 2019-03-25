# 传统匹配模型

## 匹配与排序关系

在传统的信息检索\(Information Retrieval\)领域，排序和匹配基本可以划等号：

                                     $$f(q,d)=f_{\text{BM25}}(q,d)$$ 或者 $$f(q,d)=f_{\text{LMIR}}(d|q)$$

在Web搜索中，排序和匹配是分开的，比如Learning to rank将匹配作为排序的特征：

                                      $$f(q,d)=f_{\text{BM25}}(q,d)+\text{PageRank}(d)+\cdots$$ 

![](../../../../.gitbook/assets/timline-jie-tu-20190325150250.png)

## Matching by query formulation

## Matching with term dependency

## Matching with topic model

## Matching in latent space model

![](../../../../.gitbook/assets/timline-jie-tu-20190325150506.png)

### BM25: Matching in Term Space

![](../../../../.gitbook/assets/timline-jie-tu-20190325150908.png)

### Matching in Latent Space

## Matching with translation model

![](../../../../.gitbook/assets/timline-jie-tu-20190325150539.png)

## Source

{% embed url="https://www.comp.nus.edu.sg/~xiangnan/sigir18-deep.pdf" %}

{% embed url="http://www.hangli-hl.com/uploads/3/4/4/6/34465961/wsdm\_2019\_tutorial.pdf" %}

{% embed url="http://www.hangli-hl.com/uploads/3/1/6/8/3168008/ml\_for\_match-step2.pdf" %}

