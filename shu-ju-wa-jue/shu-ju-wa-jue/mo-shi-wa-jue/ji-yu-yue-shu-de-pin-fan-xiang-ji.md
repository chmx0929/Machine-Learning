# 基于约束的频繁项集

有时我们并不关心所有的频繁项集，只想了解我们关心的。加约束条件去挖掘既灵活又可加快速度。

## 模式空间剪枝约束

#### **Anti-monotonic:** If constraint $$c$$ is violated, its further mining can be terminated

![](../../../.gitbook/assets/timline-jie-tu-20181011145016.png)

#### **Monotonic:** If $$c$$ is satisfied, no need to check $$c$$ again

![](../../../.gitbook/assets/timline-jie-tu-20181011144948.png)

#### **Succinct:** If the constraint $$c$$ can be enforced by directly manipulating the data

![](../../../.gitbook/assets/timline-jie-tu-20181011145307%20%281%29.png)

#### **Convertible:**$$c$$ can be converted to monotonic or anti-monotonic if items can be ordered in processing

![](../../../.gitbook/assets/timline-jie-tu-20181011145402.png)

## 数据空间剪枝约束

#### **Data succinct:** Data space can be pruned at the initial pattern mining process

![](../../../.gitbook/assets/timline-jie-tu-20181011145307.png)

#### **Data anti-monotonic:** If a transaction $$t$$ does not satisfy $$c$$ ,then $$t$$ can be pruned to reduce data processing effort

![](../../../.gitbook/assets/timline-jie-tu-20181011145202.png)

## 

