#include <torch/torch.h>

int main() {
  printf("---- EX1: backward ----\n");
  torch::Tensor x = torch::tensor(1.0, torch::requires_grad());
  torch::Tensor w = torch::tensor(2.0, torch::requires_grad());
  torch::Tensor b = torch::tensor(3.0, torch::requires_grad());
  auto y = w * x + b;
  y.backward();
  printf("%d\n%d\n%d\n\n", x.grad().item<int>(), w.grad().item<int>(), b.grad().item<int>());

  printf("---- EX2: Linear ----\n");
  x = torch::randn({10, 3});
  y = torch::randn({10, 2});
  torch::nn::Linear linear(3, 2);
  w = linear->weight, b = linear->bias;
  std::cout << "w:\n" << w << '\n';
  std::cout << "b:\n" << b << '\n';
  torch::nn::MSELoss criterion;
  torch::optim::SGD optimizer(linear->parameters(), torch::optim::SGDOptions(0.01));
  auto pred = linear->forward(x);
  std::cout << "pred:" << pred << '\n';
  auto loss = criterion(pred, y);
  auto loss2 = (pred - y).pow(2).sum(-1).mean(0) / 2;
  std::cout << "loss:" << loss << '\n' << loss2 << "\n\n";
  loss.backward();
  std::cout << "dL/dw:\n" << linear->weight.grad() << '\n';
  std::cout << "dL/dw(check):\n" << (x.transpose(0, 1).mm(pred - y) / x.sizes()[0]).transpose(0, 1) << '\n';
  optimizer.step();
  pred = linear->forward(x);
  loss = criterion(pred, y);
  std::cout << "loss after 1 optimization step:" << loss.item<float>() << "\n\n";

  printf("---- EX3: Create from existing array ----\n");
  float A[] = {1, 2, 3, 4};
  // from_blob 将会和A数组公用相同的地址
  torch::Tensor t1 = torch::from_blob(A, {2, 2});  // 数组, shape(必须指定)
  A[0] = 5;
  std::cout << A[0] << ' ' << t1[0][0] << '\n';
  std::cout << A << ' ' << t1.data_ptr() << '\n';  // 相同地址
  std::vector<float> V = {1, 2, 3, 4};
  torch::Tensor t2 = torch::from_blob(V.data(), {1, 4});
  std::cout << "Tensor from vector: shape=" << t2.sizes() << '\n' << t2 << '\n';

  printf("---- EX4: slice and extract part from tensor ----\n");
  auto s = torch::arange(1, 10, torch::kFloat32).view({3, 3});
  std::cout << "s:\n" << s << '\n';
  using torch::indexing::Slice;
  using torch::indexing::None;
  using torch::indexing::Ellipsis;
  std::cout << "'s[0,2]' as tensor:\n" << s.index({0, 2}) << '\n' << s[0][2] << '\n';
  std::cout << "'s[0,2]' as value:\n" << s.index({0, 2}).item<int>() << '\n';
  std::cout << "'s[:,2]':\n" << s.index({Slice(), 2}) << '\n';
  std::cout << "'s[:2]':\n" << s.index({Slice(None, 2)}) << '\n';
  std::cout << "'s[:2,:]':\n" << s.index({Slice(None, 2), Slice()}) << '\n';
  std::cout << "'s[:,1:]':\n" << s.index({Slice(), Slice(1,None)}) << '\n';
  std::cout << "'s[:,::2]':\n" << s.index({Slice(), Slice(None,None,2)}) << '\n';
  std::cout << "'s[:2,1]':\n" << s.index({Slice(None,2), 1}) << '\n';
  std::cout << "'s[...,:2]':\n" << s.index({Ellipsis, Slice(None, 2)}) << "\n\n";

  printf("---- EX5: input pipeline ----\n");
  const std::string MNIST_DATA_PATH = "../../../../data/mnist";
  auto dataset = torch::data::datasets::MNIST(MNIST_DATA_PATH)
      .map(torch::data::transforms::Normalize(0.1307, 0.3081))
      .map(torch::data::transforms::Stack());
  auto example = dataset.get_batch(0);
  std::cout << "Sample data size: " << example.data.sizes() << '\n';
  std::cout << "Sample target: " << example.target.item<int>() << '\n';
  auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    dataset, 64);
  auto& example_batch = *dataloader->begin();
  std::cout << "data size: " << example_batch.data.sizes() << '\n';
  std::cout << "target size: " << example_batch.target.sizes() << "\n\n";

  std::cout << "---- pretrained model ----\n";
  
  return 0;
}
