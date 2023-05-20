#include "ksvd.h"
#include "_omp.h"
#include "utils.h"

Eigen::MatrixXd ksvd_initation(int rows, int cols) {
   Eigen::MatrixXd D = Eigen::MatrixXd::Random(rows, cols);
   for (int i = 0; i < cols; ++i) {
        D.col(i).normalize();
   }
   return D;
}

void ksvd_update1(Eigen::MatrixXd Y, Eigen::MatrixXd &D, Eigen::MatrixXd &X) {
    int n_atoms = D.cols();   

    // update dictionary for each atom
// #pragma omp parallel for
    for (int j = 0; j < n_atoms; j++) {
        printf("ksvd atom number : %d\n", j);
        
        // find sparse nonzero entire
        Eigen::VectorXi support(X.cols());
        for (int i = 0; i < X.cols(); i++) {
            support(i) = (X(j, i) != 0) ? 1 : 0;
        }

        // if all zero, continue next atom
        if (support.sum() == 0) {
            continue;
        }

        // Y
        Eigen::MatrixXd sub_Y(Y.rows(), support.sum());
        int sub_Y_idx = 0;
        for (int i = 0; i < support.size(); ++i) {
            if (support(i) != 0) {
                sub_Y.col(sub_Y_idx++) = Y.col(i);
            }
        }
        
        // d_j
        Eigen::VectorXd atom_col = D.col(j); 

        // X_T_i
        Eigen::RowVectorXd sub_X(support.sum());
        int sub_X_idx = 0;
        for (int i = 0; i < support.size(); ++i) {
            if (support(i) != 0) {
                sub_X(sub_X_idx++) = X(j, i);
            }
        }

        // E_k = Y - d_j * X_T_i
        Eigen::MatrixXd E_k = sub_Y - atom_col * sub_X;
        
        // apply SVD in A
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(E_k, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::MatrixXd U = svd.matrixU();
        Eigen::MatrixXd V = svd.matrixV();
        Eigen::VectorXd S = svd.singularValues();

        // d_k = the first column of U
        D.col(j) = U.col(0);
        // x_k_R = the first column of V * theta(1,1)
        Eigen::VectorXd X_i = S(0,0) * V.col(0);
        int X_i_idx = 0;
        for (int i = 0; i < support.size(); ++i) {
            if (support(i) != 0) {
                X(j, i) = X_i(X_i_idx++);
            }
        }
    }
}

KsvdValues ksvd(Eigen::MatrixXd Y, int ATOM_NUM ,int ITER_NUM, int SPARSITY) {
    
    // step 1 : init dictionary
    std::cout << "ksvd init..." << std::endl;
    Eigen::MatrixXd D = ksvd_initation(Y.rows(), ATOM_NUM);
    matrix_data(D, "dictionary_original", 256);

    Eigen::MatrixXd X(ATOM_NUM, Y.cols());
    // X.setZero();

    // step 4 : guide if satisfy with stop rule
    std::cout << "ksvd train..." << std::endl;
    for (int i = 0; i < ITER_NUM; ++i) {
        
        std::cout << "train iter time: " << i << std::endl;
        
        // step 2 : sparse coding
        std::cout << "omp start..." << std::endl;
        _omp1(Y, D, X, SPARSITY);
        
        // step 3 ：update dictionary
        std::cout << "kvd update..." << std::endl;
        ksvd_update1(Y, D, X);
    }

    return {D, X};
}