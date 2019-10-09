# 搜索

## 搜索架构

搜索引擎的总体架构如下图，根据用户的特定需求返回相应的信息，即有特定query：

![](../../../.gitbook/assets/timline-jie-tu-20190325143624.png)

![](../../../.gitbook/assets/timline-jie-tu-20190325143827.png)

最为广泛的应用即网站搜索引擎，如Google、Bing、百度等，总体流程如下：

1. 第一步：先爬取线上各种网站，进行内容理解后建立索引放入存储。
2. 第二步：用户输入query，进行query理解后进行query-doc匹配从存储中获取内容之后进行排序推送给用户。

![](../../../.gitbook/assets/timline-jie-tu-20190325144828.png)

## Source

{% embed url="https://zhuanlan.zhihu.com/p/45849695" %}

{% embed url="https://www.comp.nus.edu.sg/~xiangnan/sigir18-deep.pdf" %}

{% embed url="https://zhuanlan.zhihu.com/p/58160982" %}

{% embed url="http://www.hangli-hl.com/uploads/3/4/4/6/34465961/wsdm\_2019\_tutorial.pdf" %}

