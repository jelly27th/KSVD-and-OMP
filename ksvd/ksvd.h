#ifndef _KSVD_H
#define _KSVD_H

#include "utils.h"

Eigen::MatrixXd ksvd_iniitation(Eigen::MatrixXd Y, int ATOM_NUM);
void ksvd_update(Eigen::MatrixXd Y, Eigen::MatrixXd &D, Eigen::MatrixXd &X);
void ksvd_update1(Eigen::MatrixXd Y, Eigen::MatrixXd &D, Eigen::MatrixXd &X);

#endif