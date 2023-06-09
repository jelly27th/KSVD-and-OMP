#ifndef _KSVD_H
#define _KSVD_H

#include "utils.h"

typedef struct KsvdValues {
    Eigen::MatrixXd D;
    Eigen::MatrixXd X;
} KsvdValues;

Eigen::MatrixXd ksvd_initation(int rows, int cols);
KsvdValues ksvd(Eigen::MatrixXd Y, int ATOM_NUM ,int ITER_NUM, int SPARSITY);
void ksvd_update(Eigen::MatrixXd Y, Eigen::MatrixXd &D, Eigen::MatrixXd &X);

#endif