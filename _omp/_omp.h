#ifndef __OMP_H
#define __OMP_H

#include "utils.h"

void _omp(Eigen::MatrixXd Y, Eigen::MatrixXd &D, Eigen::MatrixXd &X, int sparsity);
void _omp1(Eigen::MatrixXd Y, Eigen::MatrixXd &D, Eigen::MatrixXd &X, int sparsity);

#endif