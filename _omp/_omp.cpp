#include "_omp.h"

void _omp(Eigen::MatrixXd Y, Eigen::MatrixXd &D, Eigen::MatrixXd &X) {
    double tolerance = 1e-6; // 稀疏编码的容差
    int max_iterations = 50; // 最大迭代次数

    int n_atoms = D.cols();   // 字典原子个数
    int n_samples = Y.cols(); // 信号个数

    // 初始化稀疏编码结果
    X.setZero();

    // 对每个信号进行稀疏编码
    for (int i = 0; i < n_samples; i++)
    {
        std::cout << "omp atom number : " << i << std::endl;

        Eigen::VectorXd y = Y.col(i);          // 取出当前信号
        Eigen::VectorXd residual = y;          // 初始化残差
        Eigen::VectorXd atom_indices(n_atoms); // 初始化选中的原子下标
        atom_indices.setConstant(-1);

        int k = 0; // 已选中的原子个数

        // 迭代直到残差足够小或已选中足够多的原子
        while (residual.norm() > tolerance && k < max_iterations)
        {
            // std::cout << "omp iter time: " << k << std::endl;

            double max_correlation = 0;
            int max_index = -1;

            // 在字典中寻找与残差具有最大相关性的原子
            for (int j = 0; j < n_atoms; j++)
            {
                if (atom_indices(j) != -1)
                { // 如果该原子已被选中，则跳过
                    continue;
                }

                double correlation = std::abs(D.col(j).dot(residual));
                if (correlation > max_correlation)
                {
                    max_correlation = correlation;
                    max_index = j;
                }
            }

            // 将与残差具有最大相关性的原子加入已选中的原子集合
            atom_indices(k) = max_index;
            k++;

            // 更新稀疏编码结果
            Eigen::MatrixXd sub_D = D.leftCols(k);
            Eigen::MatrixXd sub_X = sub_D.colPivHouseholderQr().solve(y);
            X.block(0, i, k, 1) = sub_X;

            // 计算新的残差
            residual = y - sub_D * sub_X;
        }
    }
}