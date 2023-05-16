正交匹配追踪（Orthogonal Matching Pursuit，简称OMP）是一种用于稀疏信号重构的迭代算法。其基本思想是通过选择与当前残差具有最大内积的原子来逐步逼近原始信号的稀疏表示。下面是 OMP 算法的详细原理：

假设 $\mathbf{y}$ 是待重构的信号，$\mathbf{D}$ 是给定的字典矩阵， $\mathbf{x}$ 是 $\mathbf{y}$ 在 $\mathbf{D}$ 基下的系数向量，$k$ 是稀疏度，$\epsilon$ 是迭代停止的阈值。OMP 算法的流程如下：

1. 初始化：令 $\mathbf{r}=\mathbf{y}$，$\mathcal{S}=\emptyset$，$\mathbf{x}=\mathbf{0}$。

2. 选择原子：在 $\mathbf{D}$ 中寻找与 $\mathbf{r}$ 具有最大内积的原子，即 $\mathbf{d}_i=\arg\max_{\mathbf{d}_j\in\mathbf{D}}|\langle\mathbf{r}, \mathbf{d}_j\rangle|$，把它加入到集合 $\mathcal{S}$ 中。

3. 解线性方程：对集合 $\mathcal{S}$ 中的原子，求解线性方程 $\mathbf{x}_\mathcal{S}=\arg\min\|\mathbf{y}-\mathbf{D}_\mathcal{S}\mathbf{x}_\mathcal{S}\|_2$，其中 $\mathbf{D}_\mathcal{S}$ 是 $\mathbf{D}$ 中与 $\mathcal{S}$ 中的原子构成的子矩阵。

4. 更新残差：令 $\mathbf{r}=\mathbf{y}-\mathbf{D}_\mathcal{S}\mathbf{x}_\mathcal{S}$。

5. 判断终止条件：如果满足 $\|\mathbf{r}\|_2<\epsilon$ 或者 $\|\mathbf{x}_\mathcal{S}\|_0=k$，则停止迭代，输出系数向量 $\mathbf{x}$；否则，返回步骤 2。

其中，$\|\cdot\|_2$ 表示 $\ell_2$ 范数，即向量的欧几里得长度；$\|\cdot\|_0$ 表示 $\ell_0$ 范数，即向量中非零元素的个数。由于 $\ell_0$ 范数是 NP 难问题，因此通常采用 $\ell_1$ 范数来近似求解，即将步骤 3 中的 $\ell_2$ 范数最小化问题转化为 $\ell_1$ 范数最小化问题。