#include "utils.h"

cv::Mat read_image(std::string filename) {
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    cv::Mat image_double;
    image.convertTo(image_double, CV_64F);

    return image_double;
}

Eigen::MatrixXd image_to_patches(cv::Mat image, int PATCH_SIZE) {

    // 计算块的行数和列数
    int rows = image.rows / PATCH_SIZE;
    int cols = image.cols / PATCH_SIZE;

    // 创建输出矩阵
    Eigen::MatrixXd patches(PATCH_SIZE * PATCH_SIZE, rows * cols);

    // 将图像分割为块
    int patch_idx = 0;
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            // 获取当前块的 ROI
            cv::Rect roi(c * PATCH_SIZE, r * PATCH_SIZE, PATCH_SIZE, PATCH_SIZE);
            cv::Mat patch = image(roi);

            // 将块展开为列向量，并存储到输出矩阵中
            Eigen::Map<Eigen::RowVectorXd> patch_vec(patch.ptr<double>(), PATCH_SIZE * PATCH_SIZE);
            patches.col(patch_idx) = patch_vec;
            patch_idx++;
        }
    }

    return patches;
}

Eigen::MatrixXd multipy(Eigen::MatrixXd D, Eigen::MatrixXd X) {
    return D * X;
}

cv::Mat patches_to_image(Eigen::MatrixXd patches, int PATCH_SIZE) {
    int n_patches = patches.cols();                     // 块数
    int image_size = std::sqrt(n_patches) * PATCH_SIZE; // 图像大小
    cv::Mat image(image_size, image_size, CV_64FC1);    // 创建一个双精度浮点型矩阵

    // 将每个块放入图像中
    int index = 0;
    for (int i = 0; i < image_size; i += PATCH_SIZE)
    {
        for (int j = 0; j < image_size; j += PATCH_SIZE)
        {
            // 取出当前块并将其复制到图像中
            cv::Mat patch(PATCH_SIZE, PATCH_SIZE, CV_64FC1, patches.row(index).data());
            cv::Mat patch_transpose;
            cv::transpose(patch, patch_transpose);
            patch_transpose.copyTo(image(cv::Rect(j, i, PATCH_SIZE, PATCH_SIZE)));
            index++;
        }
    }
    
    cv::Mat image_uint;
    image.convertTo(image_uint, CV_8U);
    
    return image_uint;
}