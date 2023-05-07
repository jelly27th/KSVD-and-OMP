#include "utils.h"

cv::Mat read_image(std::string filename) {
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    cv::Mat image_double;
    image.convertTo(image_double, CV_64F);

    return image_double;
}

Eigen::MatrixXd image_to_patches(cv::Mat image, int PATCH_SIZE) {

    int rows = image.rows;
    int cols = image.cols;
    int n_channels = image.channels();

    // 计算图像可以被分为多少个8*8的块
    int n_rows = rows / PATCH_SIZE;
    int n_cols = cols / PATCH_SIZE;

    // 将图像转换为double类型的矩阵
    cv::Mat double_image;
    // image.convertTo(double_image, CV_64F, 1.0 / 255);
    image.convertTo(double_image, CV_64F);

    Eigen::MatrixXd patches(n_channels * PATCH_SIZE * PATCH_SIZE, n_rows * n_cols);
    Eigen::VectorXd col(PATCH_SIZE * PATCH_SIZE);
    
    int index = 0;
    for (int i = 0; i < rows; i += PATCH_SIZE ) {
        for (int j = 0; j < cols; j += PATCH_SIZE) {

            col.setZero();
            int idx = 0;
            for (int x = i; x < i + PATCH_SIZE; x++) {
                for (int y = j; y < j + PATCH_SIZE; y++) {
                    col(idx++) = double_image.at<double>(x, y);
                }
            }
            patches.col(index++) = col;
        }
    }

    

    return patches;
}

Eigen::MatrixXd multipy(Eigen::MatrixXd D, Eigen::MatrixXd X) {
    return D * X;
}

cv::Mat patches_to_image(Eigen::MatrixXd patches, int PATCH_SIZE) {
    // 计算图像块数量
    int num_patches = patches.cols();

    // 创建输出图像
    cv::Mat image(PATCH_SIZE * sqrt(num_patches), PATCH_SIZE * sqrt(num_patches), CV_8UC1);

    // 将所有图像块放置到输出图像中
    for (int i = 0; i < num_patches; i++)
    {
        // 获取当前图像块
        Eigen::MatrixXd patch_data = patches.col(i);

        // 将图像块数据复制到cv::Mat对象中
        cv::Mat patch(PATCH_SIZE, PATCH_SIZE, CV_8UC1);
        for (int j = 0; j < PATCH_SIZE; j++)
        {
            for (int k = 0; k < PATCH_SIZE; k++)
            {
                patch.at<uint8_t>(j, k) = static_cast<uint8_t>(patch_data(j * PATCH_SIZE + k));
            }
        }

        // 将当前图像块放置到输出图像中
        int row_pos = (i / sqrt(num_patches)) * PATCH_SIZE;
        int col_pos = (i % (int)sqrt(num_patches)) * PATCH_SIZE;
        cv::Rect roi(col_pos, row_pos, PATCH_SIZE, PATCH_SIZE);
        patch.copyTo(image(roi));
    }

    return image;
}

void mat_data(cv::Mat data, std::string name)
{
    name = "../data/" + name;
    std::ofstream file(name);
    file << format(data, cv::Formatter::FMT_NUMPY) << std::endl;
    file.close();
}

void matrix_data(Eigen::MatrixXd data, std::string name)
{
    name = "../data/" + name;
    std::ofstream file(name);
    Eigen::IOFormat fmt(1024, 0, ", ", "\n", "[", "]");
    file << data.format(fmt) << std::endl;
    file.close();
}
