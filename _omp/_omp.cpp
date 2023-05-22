#include "_omp.h"
#include "utils.h"

void _omp(Eigen::MatrixXd Y, Eigen::MatrixXd &D, Eigen::MatrixXd &X, int sparsity) {
    double tolerance = 1e-6; 

    int n_atoms = D.cols(); 
    int n_samples = Y.cols(); 

    // init sparse matrix `X`    
    X.setZero();
    Eigen::MatrixXd D_norm = matrix_norm(D);

    // sparse coding for each singals
// #pragma omp parallel for
    for (int i = 0; i < n_samples; i++) {

        printf("omp atom number : %d\n", i);
        
        Eigen::VectorXd y = Y.col(i);          // current singal
        Eigen::VectorXd residual = y;          // init residual
        
        Eigen::VectorXd atom_indices(n_atoms); 
        atom_indices.setConstant(-1);
        std::list<int> indices;

        int k = 0; // Number of atoms selected

        // step 5 : guide if satisfy with stop rule
        while (residual.norm() > tolerance && k < sparsity) {
            
            double max_correlation = 0;
            int max_index = -1;

            // step 1 : find Maximum correlation atom with residual
            for (int j = 0; j < n_atoms; j++) {
                
                if (atom_indices(j) != -1) { 
                    continue;
                }

                double correlation = std::abs(residual.transpose().dot(D_norm.col(j)));
                if (correlation > max_correlation) {
                    max_correlation = correlation;
                    max_index = j;
                }
            }

            // Add the atoms with the greatest correlation to the selected set of atoms
            atom_indices(max_index) = max_index;
            indices.push_back(max_index);

            // step 3 : Using Least Squares to calculate sparse matrix 
            // A_new
            Eigen::MatrixXd sub_D(D.rows(), k + 1);
            int sub_D_idx = 0;
            for (std::list<int>::iterator it = indices.begin(); it != indices.end(); it++) {
                sub_D.col(sub_D_idx++) = D.col(*it);
            }

            // L_p = (A_new)^+ * y
            Eigen::MatrixXd tmp = sub_D.transpose() * sub_D;
            Eigen::VectorXd sub_X = (tmp.inverse() * sub_D.transpose()) * y;

            // X_rec
            int sub_X_idx = 0;
            for (std::list<int>::iterator it = indices.begin(); it != indices.end(); it++) {
                X.col(i)[*it] = sub_X[sub_X_idx++];
            }
            
            // step 4: update residual
            // r = y -A_new * L_p
            residual = y - sub_D * sub_X;

            k++;
        }

        indices.clear();
    }
}




void _omp1(Eigen::MatrixXd Y, Eigen::MatrixXd &D, Eigen::MatrixXd &X, int sparsity) {
    double tolerance = 1e-6;

    int n_atoms = D.cols();
    int n_samples = Y.cols();

    // init sparse matrix `X`
    X.setZero();

    // sparse coding for each singals
    // #pragma omp parallel for
    for (int i = 0; i < n_samples; i++)
    {

        printf("omp atom number : %d\n", i);

        Eigen::VectorXd y = Y.col(i); // current singal
        Eigen::VectorXd residual = y; // init residual
        Eigen::VectorXd atom_indices(n_atoms);
        atom_indices.setConstant(-1);

        int k = 0; // Number of atoms selected

        // step 5 : guide if satisfy with stop rule
        while (residual.norm() > tolerance && k < sparsity)
        {

            double max_correlation = 0;
            int max_index = -1;

            // step 1 : find Maximum correlation atom with residual
            for (int j = 0; j < n_atoms; j++)
            {
                if (atom_indices(j) != -1)
                {
                    continue;
                }

                double correlation = std::abs(residual.transpose().dot(D.col(j)));
                if (correlation > max_correlation)
                {
                    max_correlation = correlation;
                    max_index = j;
                }
            }

            // Add the atoms with the greatest correlation to the selected set of atoms
            atom_indices(max_index) = max_index;

            // step 3 : Using Least Squares to calculate sparse matrix
            // A_new
            Eigen::MatrixXd sub_D(D.rows(), k + 1);
            int sub_D_idx = 0;
            for (int idx = 0; idx < n_atoms; idx++)
            {
                if (atom_indices[idx] != -1)
                {
                    sub_D.col(sub_D_idx++) = D.col(idx);
                }
            }

            // L_p = (A_new)^+ * y
            Eigen::MatrixXd tmp = sub_D.transpose() * sub_D;
            Eigen::VectorXd sub_X = (tmp.inverse() * sub_D.transpose()) * y;

            // X_rec
            int sub_X_idx = 0;
            for (int idx = 0; idx < n_atoms; idx++)
            {
                if (atom_indices[idx] != -1)
                {
                    X.col(i)[idx] = sub_X[sub_X_idx++];
                }
            }

            // step 4: update residual
            // r = y -A_new * L_p
            residual = y - sub_D * sub_X;

            k++;
        }
    }
}