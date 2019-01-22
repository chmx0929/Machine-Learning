# 连续概率

## 和&积&贝叶斯\(Sum&Product&Bayes rule\)

Sum： $$P(x) = \int P(x,y)dy$$ 

Product： $$P(x,y) = P(y|x)P(x)$$ 

Bayes： $$P(x|y)=\frac{P(y|x)P(x)}{P(y)}$$ 

## 常规性质\(properties\)

总概率为1： $$\int_{-\infty}^{\infty} P(x) = 1$$ 

概率为非负： $$P(x) \in \mathbb{R}  \ \ \ and \ \ \ P(x) \geq 0 $$ 

## 常用运算

期望\(Expectation\)： $$E(\int (x)) = \int P(x)f(x)dx$$ 

方差\(Variance\)： $$var(f(x)) = E[(f(x)-E[f(x)])^2] = E[f(x)^2]-E[f(x)]^2$$ 

协方差\(Covariance\)： $$cov[x,y] = E_{x,y}(xy) - E(x)E(y)$$ 

