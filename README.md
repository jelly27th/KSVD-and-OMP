Orthogonal Matching Pursuit (OMP) 算法是一种基于贪心策略的迭代算法，用于从测量数据中恢复稀疏信号的系数。下面是 OMP 算法的详细步骤：

1. 初始化残差 $r = y$，系数矩阵 $X$ 和支持集 $T = \emptyset$。

2. 选择一个原子 $D_j$，使其与残差的内积 $|D_j^T r|$ 最大，即 $j = \arg\max_{i \notin T} |D_i^T r|$。

3. 将 $j$ 加入支持集 $T$，并将选中的列向量 $D_j$ 存储在矩阵 $A_T$ 的第 $i$ 列。

4. 解决最小二乘问题 $\min_{x_T} \|y - A_T x_T\|_2^2$，得到系数向量 $x_T$。

5. 如果 $|T| = k$，则停止迭代，否则返回步骤 2。

6. 根据 $x_T$ 重构信号的系数 $x$，其中 $x_i = 0$ 对于 $i \notin T$。

7. 根据系数矩阵 $X$ 重构信号 $x$。

8. 返回重构的信号 $x$。

在实际应用中，OMP 算法可能需要进行一些调整，例如添加阈值来避免噪声的影响。此外，如果信号具有某些结构，例如稀疏表示在某个基函数上具有低秩性，那么可以使用其他算法来进一步提高稀疏表示的效果。