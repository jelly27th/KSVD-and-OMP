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

// ===========================================
// + input: two-dimensional M*N signal Y
//          M*K dictionary matrix D
//          K*N sparse matrix X
// ===========================================
void ksvd_update(Eigen::MatrixXd Y, Eigen::MatrixXd &D, Eigen::MatrixXd &X) {
    int n_atoms = D.cols();   

    // update dictionary for each atom
    for (int j = 0; j < n_atoms; j++) {
        printf("ksvd atom number : %d\n", j);
        
        // find sparse nonzero entire
        std::list<int> indices;
        for (int i = 0; i < X.cols(); i++) {
            if (X(j, i) != 0)
                indices.push_back(i);
        }

        // if all zero, continue next atom
        if (indices.size() == 0) {
            continue;
        }

        // Y
        Eigen::MatrixXd sub_Y(Y.rows(), indices.size());
        int sub_Y_idx = 0;
        for (std::list<int>::iterator it = indices.begin(); it != indices.end(); it++) {
            sub_Y.col(sub_Y_idx++) = Y.col(*it);
        }
        
        // d_j
        Eigen::VectorXd atom_col = D.col(j); 

        // X_T_i
        Eigen::RowVectorXd sub_X(indices.size());
        int sub_X_idx = 0;
        for (std::list<int>::iterator it = indices.begin(); it != indices.end(); it++) {
            sub_X(sub_X_idx++) = X(j, *it);
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
        // x_k_R = theta(1,1) * the first column of V
        Eigen::VectorXd X_i = S(0,0) * V.col(0);
        int X_i_idx = 0;
        for (std::list<int>::iterator it = indices.begin(); it != indices.end(); it++) {
            X(j, *it) = X_i(X_i_idx++);
        }

        indices.clear();
    }
}

KsvdValues ksvd(Eigen::MatrixXd Y, int ATOM_NUM ,int ITER_NUM, int SPARSITY) {
    
    // step 1 : init dictionary
    std::cout << "ksvd init..." << std::endl;
    Eigen::MatrixXd D = ksvd_initation(Y.rows(), ATOM_NUM);

    Eigen::MatrixXd X(ATOM_NUM, Y.cols());

    // step 4 : guide if satisfy with stop rule
    std::cout << "ksvd train..." << std::endl;
    for (int i = 0; i < ITER_NUM; ++i) {
        
        std::cout << "train iter time: " << i << std::endl;
        
        // step 2 : sparse coding
        std::cout << "omp start..." << std::endl;
        _omp(Y, D, X, SPARSITY);
        
        // step 3 ：update dictionary
        std::cout << "kvd update..." << std::endl;
        ksvd_update(Y, D, X);
    }

    return {D, X};
}
