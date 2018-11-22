# 中心度分析

中心度\(Centrality\): How "central" a node is in the network

## 基础衡量指标

### **Degree centrality**

    ****degree of a node \(the higher degree, more important the node\)

### **Eccentricity centrality**

    the less eccentric, the more central

        $$c(v_i) = 1/e(v_i)$$ ; Central node: $$e(v_i) = r(G)$$ \(if it equals the radius of G\)

        Periphery node: $$e(v_i) = d(G)$$ \(if it equals the diameter of G\)

### **Closeness centrality**

    ****the average of the shortest path length from the node to every other node in the network, indicating how close a node: $$c(v_i) = 1/\sum_j d(v_i,v_j)$$

        median node $$v_m$$ if $$v_m$$ has the smallest total distance $$\sum_j d(v_m,v_j)$$ 

### Betweenness centrality

$$\#$$ of shortest paths from all vertices to all others that pass through $$v$$: 

$$c(v_i) = \sum \limits_{j\neq i} \sum \limits_{k\neq i, k\gt j} \frac{\eta_{jk}(v_i)}{\eta_{jk}}$$ , $$\eta_{jk}$$:$$\#$$ of shortest paths between $$v_j$$ and $$v_k$$, $$\eta_{jk}(v_i)$$:$$\#$$ paths contain $$v_i$$

### Eigenvector centrality

Measure the influence of a node in a network, i.e., connections to high-scoring nodes contribute more to the score of the node in question than equal connections to low-scoring nodes

## Web网络中心度衡量

### 有向图

对于节点 $$v$$ ：越多的节点指向 $$v$$ 则它的声望越高；越多高声望节点指向 $$v$$ 则它的声望越高

节点声望\(prestige\)： $$p(v) = \sum \limits_u A(u,v)\cdot p(u) = \sum \limits_u A^T(v,u)\cdot p(u)$$ 

可写成： $$p' = A^Tp$$ ，经过 $$k$$ 轮迭代后，我们得到 $$p_k = (A^T)^kp_0$$。随着 $$k$$ 的增加，向量 $$p_k$$ 收敛

### PageRank

PageRank让链接来“投票”。一个页面的“得票数”由所有链向它的页面的重要性来决定，到一个页面的超链接相当于对该页投一票。一个页面的PageRank是由所有链向它的页面的重要性经过递归算法得到的。一个有较多链入的页面会有较高的登记，相反如果一个页面没有任何链入，那么它没有等级\(后来做平滑，从高等级页面中取出一部分声望匀给这些，保证平滑\)

例子：假设有 $$A,B,C$$ 和 $$D$$。

若所有页面都链向$$A$$，那么$$A$$的PR\(PageRank\)值将是$$B,C$$和$$D$$的PageRank总和：

                                       $$PR(A)=PR(B)+PR(C)+PR(D)$$ 

 $$B$$ 也有链向 $$C$$；并且 $$D$$ 也有链接到 $$A,B,C$$  三个页面。一个页面不能投票2次。所以 $$B$$ 给每个页面半票， $$D$$ 给每个页面三分之一票：

                                            $$PR(A) = \frac{PR(B)}{2}+\frac{PR(C)}{1}+\frac{PR(D)}{3}$$ 

即                                       $$PR(A) = \frac{PR(B)}{L(B)}+\frac{PR(C)}{L(C)}+\frac{PR(D)}{L(D)}$$ 

又做了平滑，即拿出一部分声望 $$\frac{1-d}{N}$$ 来匀给从未出现过的页面：

                                   $$PR(A) = (\frac{PR(B)}{L(B)}+\frac{PR(C)}{L(C)}+\frac{PR(D)}{L(D)}+\cdots)d+\frac{1-d}{N}$$ 

如果给每个页面一个随机PageRank值\(非0\)，那么经过不断地迭代计算，这些页面的PR值会稳定，即收敛

### HITS

按照HITS算法，用户输入关键词后，算法对返回的匹配页面计算两种值，一种是枢纽值（Hub Scores），另一种是权威值\(Authority Scores\),这两种值是互相依存、互相影响的。所谓枢纽值，指的是页面上所有导出链接指向页面的权威值之和。权威值是指所有导入链接所在的页面中枢纽值之和。

                                   $$a(v)=\sum \limits_u A^T(v,u)\cdot h(u)$$             $$h(v) = \sum \limits_uA(v,u)\cdot a(v)$$ 

$$a_k = A^Th_{k-1}=A^T(Aa_{k-2})=(A^TA)a_{k-2}$$               $$h_k=Aa_{k-1}=A(A^Th_{k-2})=(AA^T)h_{k-2}$$ 

## 社交网络中的分析度量指标

**Betweenness:** The extent to which a node lies between other nodes in the network. This measure takes into account the connectivity of the node's neighbors, giving a higher value for nodes which bridge clusters. The measure reflects the number of people who a person is connecting indirectly through their direct links

**Bridge:** An edge is a bridge if deleting it would cause its endpoints to lie in different components of a graph

**Centrality:** This measure gives a rough indication of the social power of a node based on how well they "connect" the network. "Betweenness", "Closeness", and "Degree" are all measures of centrality

**Centralization:** The difference between the number of links for each node divided by maximum possible sum of differences. A centralized network will have many of its links dispersed around one or a few nodes, while a decentralized network is one in which there is little variation between the number of links each node possesses

**Closeness:** The degree an individual is near all other individuals in a network \(directly or indirectly\). It reflects the ability to access information through the "grapevine" of network members. Thus, closeness is the inverse of the sum of the shortest distances between each individual and every other person in the network

**Clustering coefficient:** A measure of the likelihood that two associates of a node are associates themselves. A higher clustering coefficient indicates a greater 'cliquishness'

**Cohesion:** The degree to which actors are connected directly to each other by cohesive bonds. Groups are identified as ‘cliques’ if every individual is directly tied to every other individual, ‘social circles’ if there is less stringency of direct contact, which is imprecise, or as structurally cohesive blocks if precision is wanted

**Degree \(or geodesic distance\):** The count of the number of ties to other actors in the network

**\(Individual-level\) Density:** The degree a respondent's ties know one another/ proportion of ties among an individual's nominees. Network or global-level density is the proportion of ties in a network relative to the total number possible \(sparse versus dense networks\)

**Flow betweenness centrality:** The degree that a node contributes to sum of maximum flow between all pairs of nodes \(not that node\)

**Eigenvector centrality:** A measure of the importance of a node in a network. It assigns relative scores to all nodes in the network based on the principle that connections to nodes having a high score contribute more to the score of the node in question

**Local Bridge:** An edge is a local bridge if its endpoints share no common neighbors. Unlike a bridge, a local bridge is contained in a cycle

**Path Length:** The distances between pairs of nodes in the network. Average path-length is the average of these distances between all pairs of nodes

**Prestige:** In a directed graph prestige is the term used to describe a node's centrality. "Degree Prestige", "Proximity Prestige", and "Status Prestige" are all measures of Prestige

**Radiality Degree:** an individual’s network reaches out into the network and provides novel information and influence

**Reach:** The degree any member of a network can reach other members of the network

**Structural cohesion:** The minimum number of members who, if removed from a group, would disconnect the group

**Structural equivalence:** Refers to the extent to which nodes have a common set of linkages to other nodes in the system. The nodes don’t need to have any ties to each other to be structurally equivalent

**Structural hole:** Static holes that can be strategically filled by connecting one or more links to link together other points

