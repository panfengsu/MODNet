
#include <torch/script.h> // One-stop header.
#include <opencv2/opencv.hpp>

int main(int argc, const char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "load model ok\n";

  // Create a vector of inputs.
  std::string image_path = argv[2];
  cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

  int ref_size = 512;

  // resize image for input
  int im_h = img.rows;
  int im_w = img.cols;
  int im_rh = 0;
  int im_rw = 0;

  if ((std::max(im_h, im_w) < ref_size) || (std::min(im_h, im_w) > ref_size)) {
    if (im_w >= im_h) {
      im_rh = ref_size;
      im_rw = int((float)im_w / (float)im_h * ref_size);
    }
    else if (im_w < im_h) {
      im_rw = ref_size;
      im_rh = int((float)im_h / (float)im_w * ref_size);
    }
  }
  else {
    im_rh = im_h;
    im_rw = im_w;
  }

  std::cout << im_rw << im_rh << std::endl;

  im_rw = im_rw - im_rw % 32;
  im_rh = im_rh - im_rh % 32;

  std::cout << im_rw << im_rh << std::endl;

  cv::resize(img, img, cv::Size(im_rw, im_rh));

  img.convertTo(img, CV_32FC3, 1/255.0 );

  //std::vector<double> norm_mean = {0.5, 0.5, 0.5};
  //std::vector<double> norm_std = {0.5, 0.5, 0.5};

  img = (img - 0.5) / 0.5;

  torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, c10::kFloat);

  std::cout << img.rows << img.cols << std::endl;

  img_tensor = img_tensor.permute({2, 0, 1});
  img_tensor.unsqueeze_(0);
  //img_tensor = torch::data::transforms::Normalize<>(norm_mean, norm_std)(img_tensor);

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(img_tensor);

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();

  output = output.mul(255).clamp(0, 255).to(torch::kU8);

  int frame_h = output.size(2);
  int frame_w = output.size(3);

  std::cout << frame_h << frame_w << std::endl;

  cv::Mat resultImg(frame_h, frame_w, CV_8UC1);

  std::memcpy((void *) resultImg.data, output.data_ptr(), sizeof(torch::kU8) * output.numel());

  cv::resize(resultImg, resultImg, cv::Size(im_w, im_h));

  cv::imwrite("result.png", resultImg);

}
 
