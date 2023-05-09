#include "ksvd.h"
#include "utils.h"

Eigen::MatrixXd ksvd_iniitation(Eigen::MatrixXd Y, int ATOM_NUM) {
    // 取样本前 ATOM_NUM 列作为字典
    Eigen::MatrixXd D = Y.leftCols(ATOM_NUM);

    // L2 范数归一化字典
    // for (int i = 0; i < ATOM_NUM; i++)
    // {
    //     double norm = D.col(i).norm();
    //     D.col(i) /= norm;
    // }
    // return D;

    Eigen::MatrixXd D_norm = matrix_norm(D);
    return D_norm;
}

void ksvd_update(Eigen::MatrixXd Y, Eigen::MatrixXd &D, Eigen::MatrixXd &X) {
    int n_atoms = D.cols();   // 字典原子个数
    int n_samples = Y.cols(); // 样本个数

    // 对每个原子进行更新
    for (int j = 0; j < n_atoms; j++)
    {
        std::cout << "ksvd atom number " << j << std::endl;
        
        Eigen::MatrixXd error = Y - D * X;                              // 计算当前字典下的误差
        Eigen::VectorXd atom_col = D.col(j);                            // 取出当前原子列向量
        // Eigen::VectorXi support = (X.row(j).array() != 0).select(1, 0); // 取出当前原子的支持集
        Eigen::VectorXi support(X.cols());
        for (int i = 0; i < X.cols(); i++)
        {
            support(i) = (X(j, i) != 0) ? 1 : 0;
        }

        // 如果当前原子不在任何样本的支持集中，则跳过
        if (support.sum() == 0)
        {
            continue;
        }

        // 取出当前原子的支持样本集合
        Eigen::MatrixXd sub_Y = Y.leftCols(support.sum());
        Eigen::MatrixXd sub_X = X.topRows(n_atoms).leftCols(support.sum());

        // 更新当前原子列向量
        Eigen::VectorXd new_atom_col = sub_Y * sub_X.row(j).transpose();
        new_atom_col /= new_atom_col.norm();
        D.col(j) = new_atom_col;
    }
}

void ksvd_update1(Eigen::MatrixXd Y, Eigen::MatrixXd &D, Eigen::MatrixXd &X) {
    int n_atoms = D.cols();   // 字典原子个数
    int n_samples = Y.cols(); // 样本个数

    // 对每个原子进行更新
    for (int j = 0; j < n_atoms; j++)
    {
        std::cout << "ksvd atom number " << j << std::endl;

        Eigen::MatrixXd error = Y - D * X;   // 计算当前字典下的误差
        Eigen::VectorXd atom_col = D.col(j); // 取出当前原子列向量
        // 取出当前原子的支持集
        Eigen::VectorXi support(X.cols());
        for (int i = 0; i < X.cols(); i++)
        {
            support(i) = (X(j, i) != 0) ? 1 : 0;
        }

        // 如果当前原子不在任何样本的支持集中，则跳过
        if (support.sum() == 0)
        {
            continue;
        }

        // 取出当前原子的支持样本集合
        // Eigen::MatrixXd sub_Y = Y.leftCols(support.sum());
        Eigen::MatrixXd sub_Y(Y.rows(), support.sum());
        int sub_Y_idx = 0;
        for (int i = 0; i < support.size(); ++i) {
            if (support(i) != 0) {
                sub_Y.col(sub_Y_idx++) = Y.col(i);
            }
        }

        // Eigen::MatrixXd sub_X = X.topRows(n_atoms).leftCols(support.sum());
        Eigen::RowVectorXd sub_X(support.sum());
        int sub_X_idx = 0;
        for (int i = 0; i < support.size(); ++i) {
            if (support(i) != 0) {
                sub_X(sub_X_idx++) = X(j, i);
            }
        }

        // 更新当前原子列向量
        // Eigen::VectorXd new_atom_col = sub_Y * sub_X.row(j).transpose();
        // new_atom_col /= new_atom_col.norm();
        Eigen::MatrixXd E_k = sub_Y - atom_col * sub_X;
        // 对 A 进行 SVD 分解
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(E_k, Eigen::ComputeThinU | Eigen::ComputeThinV);
        // 获取分解后的矩阵和向量
        Eigen::MatrixXd U = svd.matrixU();
        Eigen::MatrixXd V = svd.matrixV();
        Eigen::VectorXd S = svd.singularValues();

        // D.col(j) = new_atom_col;
        D.col(j) = U.col(0);
        Eigen::VectorXd X_i = S(0,0) * V.col(0);
        int X_i_idx = 0;
        for (int i = 0; i < support.size(); ++i) {
            if (support(i) != 0) {
                X(j, i) = X_i(X_i_idx++);
            }
        }
    }
}