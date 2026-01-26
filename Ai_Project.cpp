#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

// Load your ONNX model once in main() and pass it here
int predictPath(std::string imagePath, cv::dnn::Net& net) {

    // 1. Read the image
    cv::Mat img = cv::imread(imagePath);

    if (img.empty()) {
        std::cerr << "[Error] Could not read image: " << imagePath << std::endl;
        return 0;
    }

    // ---------------------------------------------------------
    // MANUAL PREPROCESSING (Fixes the Shape Mismatch Error)
    // TensorFlow models often want NHWC: [1, 224, 224, 3]
    // ---------------------------------------------------------

    // A. Resize to target_size (224, 224)
    cv::resize(img, img, cv::Size(224, 224));

    // B. Swap Colors (BGR -> RGB)
    // OpenCV reads BGR, Keras trained on RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // C. Rescale (1./255)
    // Convert integers to floats and scale to 0.0 - 1.0
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    // D. Wrap in 4D Blob [Batch, Height, Width, Channels]
    // We create a Matrix header that points to the image data with the specific shape
    // dimensions: 1 image, 224 height, 224 width, 3 channels
    int dimensions[4] = { 1, 224, 224, 3 };
    cv::Mat blob(4, dimensions, CV_32F, img.data);

    // ---------------------------------------------------------

    // 3. Feed to Network
    net.setInput(blob);

    // 4. Forward Pass
    cv::Mat output = net.forward();

    // 5. Interpret Binary Result
    // In Keras 'binary', output is a single float probability
    float probability = output.at<float>(0, 0);

    // Thresholding (Standard is 0.5)
    int predictedClass = (probability > 0.5) ? 1 : 0;

    std::cout << "Path: " << imagePath << std::endl;
    std::cout << "Probability (Class 1): " << probability << std::endl;
    std::cout << "Final Prediction: Class " << predictedClass << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    return predictedClass;
}


void denoiseImg(std::string imagePath, cv::dnn::Net& net) {
    cv::Mat img = cv::imread(imagePath);

    if (img.empty()) {
        std::cerr << "[Error] Could not read image: " << imagePath << std::endl;
        return;
    }
    cv::resize(img, img, cv::Size(224, 224));
    img.convertTo(img, CV_32F, 1.0 / 255.0);
    int dimensions[4] = { 1, 224, 224, 3 };
    cv::Mat blob(4, dimensions, CV_32F, img.data);
    net.setInput(blob);
    cv::Mat outputBlob = net.forward();
    // 9. Reshape Output
    //    The output blob is likely 4D: [1, 224, 224, 3]. 
    //    We need to map this back to a standard 2D image matrix.
    //    We construct a generic Mat using the pointer to the raw data.
    cv::Mat denoisedImg(224, 224, CV_32FC3, outputBlob.ptr<float>());
    denoisedImg.convertTo(denoisedImg, CV_8UC3, 255.0);
    cv::cvtColor(denoisedImg, denoisedImg, cv::COLOR_RGB2BGR);
    std::string outputFilename = "denoised_result.jpg";

    // Write the image to the current directory
    bool success = cv::imwrite(outputFilename, denoisedImg);

    if (success) {
        std::cout << "Success! Saved image to: " << outputFilename << std::endl;
    }
    else {
        std::cerr << "Error: Failed to save the image." << std::endl;
    }
}
int main() {
    try {
        // Load the model exported to ONNX
        // Note: Ensure classifier_model.onnx is in the same folder as your .exe
        cv::dnn::Net net = cv::dnn::readNetFromONNX("classifier_model.onnx");
        cv::dnn::Net denoiseNet = cv::dnn::readNetFromONNX("denoiser.onnx");
        // Test it
        int predictedClass = predictPath("Schematics  (2).png", net);
        if (predictedClass == 0) {
            std::cout << "This image is a photographe, applying denoising.." << std::endl;
            denoiseImg("Schematics  (2).png", denoiseNet);
        }
        std::cout << "Bert" << std::endl;
        std::cin.get();
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error: " << e.what() << std::endl;
    }
    return 0;
}