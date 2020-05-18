# 人群定向

最近负责广告人群定向\(兴趣/行业\)体系的重构，写本篇记录一下工作。

## 体系框架

调研了几个重点竞品，可以明显看出兴趣体系框架是由产品同学定的，所以或多或少的会掺杂一些主观因素。当然一个好的产品能让体系较为合理，但总有需要调整的地方，具体怎么调，就需要用数据说话了，所以我们先制定了兴趣体系框架制定原则。

<table>
  <thead>
    <tr>
      <th style="text-align:center">&#x5E8F;&#x53F7;</th>
      <th style="text-align:center">&#x539F;&#x5219;</th>
      <th style="text-align:center">&#x89E3;&#x91CA;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center">1</td>
      <td style="text-align:center">&#x9002;&#x7528;</td>
      <td style="text-align:center">
        <p>&#x529F;&#x80FD;&#x4F5C;&#x51FA;&#x6765;&#x662F;&#x7ED9;&#x5E7F;&#x544A;&#x4E3B;&#x670D;&#x52A1;&#x7684;&#xFF0C;&#x8981;&#x8D34;&#x5408;&#x5E7F;&#x544A;&#x4E3B;&#x4F7F;&#x7528;&#x4E60;&#x60EF;</p>
        <p>&#x6240;&#x4EE5;&#x80FD;&#x5BF9;&#x6807;&#x7ADE;&#x54C1;&#x7684;&#x5BF9;&#x6807;&#x7ADE;&#x54C1;&#xFF0C;&#x65B9;&#x4FBF;&#x5E7F;&#x544A;&#x4E3B;&#x591A;&#x5E73;&#x53F0;&#x8FC1;&#x79FB;</p>
      </td>
    </tr>
    <tr>
      <td style="text-align:center">2</td>
      <td style="text-align:center">&#x4EF7;&#x503C;</td>
      <td style="text-align:center">
        <p>&#x56E0;&#x4E3A;&#x662F;&#x7528;&#x505A;&#x5E7F;&#x544A;&#x4EBA;&#x7FA4;&#x5B9A;&#x5411;&#xFF0C;&#x6240;&#x4EE5;&#x5174;&#x8DA3;&#x7C7B;&#x522B;&#x8981;&#x6709;&#x5546;&#x4E1A;&#x4EF7;&#x503C;</p>
        <p>&#x4E0D;&#x9700;&#x8981;&#x505A;UGC&#x5168;&#x90E8;&#x5174;&#x8DA3;</p>
      </td>
    </tr>
    <tr>
      <td style="text-align:center">3</td>
      <td style="text-align:center">&#x4F4E;&#x91CD;&#x5408;</td>
      <td style="text-align:center">
        <p>&#x6807;&#x7B7E;&#x610F;&#x4E49;&#x660E;&#x786E;&#xFF0C;&#x4E24;&#x7C7B;&#x522B;A&#x548C;B&#xFF0C;&#x6700;&#x7EC8;A&#x2229;B&#x4EBA;&#x7FA4;&#x5360;A&#x7C7B;&#x548C;B&#x7C7B;&#x7684;&#x5404;&#x81EA;30%&#x4EE5;&#x4E0B;</p>
        <p>&#x521A;&#x5F00;&#x59CB;&#x6CA1;&#x8BA1;&#x7B97;&#x51FA;&#x4EBA;&#x7FA4;&#x4E8B;&#xFF0C;&#x53EF;&#x4EE5;&#x8BA1;&#x7B97;&#x5173;&#x952E;&#x8BCD;&#x91CD;&#x5408;&#x4F4E;&#x4E8E;30%</p>
      </td>
    </tr>
    <tr>
      <td style="text-align:center">4</td>
      <td style="text-align:center">&#x8986;&#x76D6;&#x5168;</td>
      <td style="text-align:center">&#x6BCF;&#x4E2A;&#x5E7F;&#x544A;&#x4E3B;&#x81F3;&#x5C11;&#x80FD;&#x627E;&#x5230;&#x4E00;&#x4E2A;&#x76F8;&#x5173;&#x5174;&#x8DA3;&#x7C7B;&#x522B;&#x6807;&#x7B7E;</td>
    </tr>
    <tr>
      <td style="text-align:center">5</td>
      <td style="text-align:center">&#x4ECE;&#x5C5E;&#x5408;&#x7406;</td>
      <td style="text-align:center">
        <p>&#x591A;&#x7EA7;&#x5212;&#x5206;&#x57FA;&#x4E8E;&#x7528;&#x6237;&#x7684;&#x81EA;&#x7136;&#x7406;&#x89E3;&#xFF0C;&#x901A;&#x8FC7;A&#x7C7B;&#x522B;(&#x5B50;&#x7C7B;)&#x8BCD;&#x548C;B&#x7C7B;&#x522B;(&#x7236;&#x7C7B;)&#x8BCD;</p>
        <p>&#x7684;&#x6761;&#x4EF6;&#x6982;&#x7387;p(A|B)*p(B|A)&#x8BA1;&#x7B97;</p>
      </td>
    </tr>
    <tr>
      <td style="text-align:center">6</td>
      <td style="text-align:center">&#x91CF;&#x7EA7;&#x9002;&#x4E2D;</td>
      <td style="text-align:center">&#x82E5;&#x91CF;&#x7EA7;&#x8FC7;&#x5927;&#xFF0C;&#x5C06;&#x5176;&#x5347;&#x7EA7;&#x6216;&#x8FDB;&#x884C;&#x7EC6;&#x7C92;&#x5EA6;&#x62C6;&#x5206;
        <br
        />&#x82E5;&#x7C7B;&#x522B;&#x4E0B;&#x91CF;&#x7EA7;&#x8FC7;&#x5C0F;&#x6216;&#x53EF;&#x62C6;&#x5206;&#x7C7B;&#x522B;&#x8FC7;&#x5C11;&#xFF0C;&#x8FDB;&#x884C;&#x964D;&#x7EA7;</td>
    </tr>
  </tbody>
</table>原则定下来后，再往下做就比较简单了，但是还有一些小trick，比如教育培训-兴趣培训-绘画、书法、陶艺、乐器、表演、跆拳道、游泳、烹饪、编程...分类过细人群会很少不利于投放，全都合并到教育培训-兴趣培训级别人群又会过多。

广点通的做法是兴趣培训下只有一个三级叫科技与编程，但是兴趣培训一千三百多万人，唯一的科技与编程三级有九百万人，其他的兴趣隐式的放在了兴趣培训二级下；巨量的做法\(如下图\)是合并成一个三级，但是有另外维度的切分，即下图素质教育，涵盖兴趣培训，但又限定在针对中小学生。

我们的做法借鉴了巨量，在兴趣培训二级下设置艺术类、技能类两个三级，若某一类人群依旧过多，可以拆分成四级\(体育与运动、声乐与乐器、科技与编程、美术/书法与手工艺、舞蹈与表演\)，为自己留后路，做到大类尽量不调整，调整的话通过增加子类别微调。

![&#x5DE8;&#x91CF;&#x5174;&#x8DA3;&#x5B9A;&#x5411;&#x793A;&#x4F8B;](../../../.gitbook/assets/ju-liang-xing-qu-ding-xiang.png)

当然这种主观的体系设置，每个人理解总会有偏差，比如基础教育，有人理解为学前教育，有人理解为中小学教育，有人理解为面向全年龄的各种科普...所以像上图巨量的示例展示关键词是十分必要的。但巨量采用了比较naive的方法，我简单看了一下其展示的相关词，仅是分词了广告内容作为相关词，会造成不同类别相关词重合大，意义模糊，比如上图中的培训班、暑假班。我们的做法是采用内容定向数据结合爬取的竞品关键词、微博高频词、其他专业性信息源的词，每个类别人工标注了20至50个具有区分度词帮助广告主理解释义，也可用于词匹配增加样本等。

## 数据处理

### 样本采样

#### 正样本

因为我们的最终目的，是想对应类别的广告主的广告曝光给用户后，用户会进行互动或导流转化等。所以，最理想的正样本为与此类别广告进行过互动或导流转化人群。然而这类用户非常非常少，这也是广告的难点之一。所以我们做了以下的扩充操作：

<table>
  <thead>
    <tr>
      <th style="text-align:center">&#x9014;&#x5F84;</th>
      <th style="text-align:left">&#x8BF4;&#x660E;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center">&#x5B98;&#x65B9;&#x53F7;</td>
      <td style="text-align:left">
        <p>&#x901A;&#x8FC7;&#x4E00;&#x4E9B;&#x6CA1;&#x6709;&#x6295;&#x8FC7;&#x5E7F;&#x544A;&#x4F46;&#x7C7B;&#x522B;&#x660E;&#x786E;&#x7684;&#x5B98;&#x65B9;&#x8D26;&#x53F7;&#xFF0C;&#x5176;&#x7C89;&#x4E1D;&#x4F5C;&#x4E3A;&#x6F5C;&#x5728;&#x6B63;&#x6837;&#x672C;&#x96C6;</p>
        <p>eg.&#x5047;&#x8BBE;&#x675C;&#x857E;&#x65AF;&#x6CA1;&#x6295;&#x5E7F;&#x544A;&#xFF0C;&#x4F46;&#x5176;&#x4E3A;&#x65E5;&#x7528;&#x767E;&#x8D27;&#x7C7B;&#x522B;&#xFF0C;&#x5176;&#x7C89;&#x4E1D;&#x4E3A;&#x6F5C;&#x5728;&#x6B63;&#x6837;&#x672C;</p>
      </td>
    </tr>
    <tr>
      <td style="text-align:center">&#x5173;&#x952E;&#x8BCD;</td>
      <td style="text-align:left">
        <p>&#x901A;&#x8FC7;&#x5404;&#x7C7B;&#x522B;&#x6709;&#x533A;&#x5206;&#x5EA6;&#x7684;&#x5173;&#x952E;&#x8BCD;&#x4E0E;&#x7C7B;&#x522B;&#x6620;&#x5C04;&#xFF0C;&#x8FDB;&#x884C;&#x8BCD;&#x5339;&#x914D;&#x7B49;&#x589E;&#x52A0;&#x6F5C;&#x5728;&#x6B63;&#x6837;&#x672C;</p>
        <p>eg.&#x8FD9;&#x79CD;&#x505A;&#x6CD5;&#x6BD4;&#x8F83;&#x5E38;&#x89C4;&#xFF0C;&#x5C31;&#x4E0D;&#x4E3E;&#x4F8B;&#x4E86;</p>
      </td>
    </tr>
    <tr>
      <td style="text-align:center">&#x76F4;&#x64AD;&#x95F4;</td>
      <td style="text-align:left">
        <p>&#x901A;&#x8FC7;&#x76F4;&#x64AD;&#x95F4;&#x7684;&#x76F4;&#x64AD;&#x5185;&#x5BB9;&#xFF0C;&#x786E;&#x5B9A;&#x76F4;&#x64AD;&#x95F4;&#x4E0E;&#x7C7B;&#x522B;&#x5173;&#x7CFB;&#xFF0C;&#x5176;&#x89C2;&#x4F17;&#x4F5C;&#x4E3A;&#x6F5C;&#x5728;&#x6B63;&#x6837;&#x672C;</p>
        <p>eg.&#x8003;&#x7814;&#x6570;&#x5B66;&#x8F85;&#x5BFC;&#x76F4;&#x64AD;&#x95F4;&#x7684;&#x89C2;&#x4F17;&#x4F5C;&#x4E3A;&#x8003;&#x7814;&#x57F9;&#x8BAD;&#x7684;&#x6F5C;&#x5728;&#x6B63;&#x6837;&#x672C;</p>
      </td>
    </tr>
    <tr>
      <td style="text-align:center">LBS</td>
      <td style="text-align:left">
        <p>&#x901A;&#x8FC7;LBS&#x7535;&#x5B50;&#x56F4;&#x680F;&#x5B9A;&#x4F4D;&#x7279;&#x6B8A;&#x4F4D;&#x7F6E;&#x4E0A;&#x7EBF;&#x7684;&#x8D26;&#x53F7;&#x4F5C;&#x4E3A;&#x6F5C;&#x5728;&#x6B63;&#x6837;&#x672C;&#x96C6;</p>
        <p>eg.&#x4E07;&#x901A;&#x6C7D;&#x4FEE;&#x5B66;&#x6821;&#x56F4;&#x680F;&#x5185;&#x591A;&#x6B21;&#x4E0A;&#x7EBF;&#x7684;&#x7528;&#x6237;&#x4F5C;&#x4E3A;&#x6C7D;&#x4FEE;&#x57F9;&#x8BAD;&#x6F5C;&#x5728;&#x6B63;&#x6837;&#x672C;</p>
      </td>
    </tr>
    <tr>
      <td style="text-align:center">APP</td>
      <td style="text-align:left">
        <p>&#x901A;&#x8FC7;&#x7528;&#x6237;&#x624B;&#x673A;&#x4E0A;&#x7684;APP&#x5217;&#x8868;&#x786E;&#x5B9A;&#x8D26;&#x6237;&#x4E3A;&#x67D0;&#x7C7B;&#x522B;&#x6F5C;&#x5728;&#x6B63;&#x6837;&#x672C;</p>
        <p>eg.&#x5B89;&#x88C5;&#x4E86;&#x6C7D;&#x8F66;&#x8FDD;&#x7AE0;&#x67E5;&#x8BE2;APP&#x7684;&#x4F5C;&#x4E3A;&#x6C7D;&#x8F66;&#x4FDD;&#x517B;&#x7684;&#x6F5C;&#x5728;&#x6B63;&#x6837;&#x672C;</p>
      </td>
    </tr>
  </tbody>
</table>#### 负样本



### 特征处理

我们先设想一个比较完美的特征 Embedding 分配方案，如果它存在，应该是这个样子的：对于高频出现的特征，能够分配给它较长的 Embedding 大小，使其能更充分地编码和表达信息。而对于低频的特征，则希望分配较短的 Embedding，因为对于低频特征，它在训练数据中出现次数少，如果分配了较长的 Embedding，更容易出现过拟合现象，影响模型泛化性能。而对于那些极低频的特征，基本学不了什么知识，反而会带来各种噪音，那么我们可以不分配或者让它们共享一个公有 Embedding 即可。

## 算法模型



## 相关任务

### 冷启动

对于冷启动问题，即，新用户，身上无任何标签，但我们也要通过人群定向给他曝光广告。给他打什么标签？曝光什么广告？这里我们基于模拟退火思想，制定了一系列冷启动解决方案。

思路简单来说即先试探性给用户打上兴趣标签，根据其对相应UGC内容和广告的反馈，调节之前的兴趣标签。所以核心点就是两个: 1、试探策略怎样制定；2、反馈调节怎样制定。

#### 试探策略

#### 反馈调节



