#include <iostream>
#include <tuple>
extern "C"
{
#include <check.h>
#include <buffer.h>
#include <tensor.h>
#include <view.h>
#include <errors.h>
#include <datatype.h>
#include <optimizer.h>
#include <layer.h>
#include <function.h>
#include <test_helper.h>
#include <cost.h>
}
#include <test_helper_torch.h>

#define STOCASTIC_GRADIENT_DESCENT_CASES 10
#define RMS_PROP_CASES 6
#define ADAM_CASES 5

float32_t sgd_learning_rate_f[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    1e-3,
    1e-1,
    1e-3,
    1e-1,
    1e-3,
    1e-3,
    1e-3,
    1e-3,
    1e-3,
    1e-1,
};

float64_t sgd_learning_rate[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    1e-3,
    1e-1,
    1e-3,
    1e-1,
    1e-3,
    1e-3,
    1e-3,
    1e-3,
    1e-3,
    1e-1,
};

float32_t sgd_momentum_f[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.9,
    0.9,
    0.8,
    0.8,
    0.1,
    0.9,
};

float64_t sgd_momentum[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.9,
    0.9,
    0.8,
    0.8,
    0.1,
    0.9,
};

float32_t sgd_dampening_f[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    0.0,
    0.0,
    0.0,
    0.1,
    0.0,
    0.2,
    0.0,
    0.0,
    0.0,
};

float64_t sgd_dampening[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.1,
    0.0,
    0.2,
    0.0,
    0.0,
    0.0,
};

float32_t sgd_weight_decay_f[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    0.0,
    0.0,
    1e-1,
    1e-1,
    0.0,
    1e-1,
    2e-1,
    1e-3,
    0.0,
    1e-1,
};

float64_t sgd_weight_decay[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    0.0,
    0.0,
    1e-1,
    1e-1,
    0.0,
    1e-1,
    2e-1,
    1e-3,
    0.0,
    1e-1,
};

bool sgd_nesterov[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    false,
    false,
    false,
    false,
    false,
    false,
    false,
    true,
    true,
    true,
};

int sgd_iterations[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
};

float32_t rms_prop_learning_rate_f[RMS_PROP_CASES] = {
    1e-3,
    1e-2,
    1e-3,
    1e-2,
    1e-3,
    1e-2,
};

float64_t rms_prop_learning_rate[RMS_PROP_CASES] = {
    1e-3,
    1e-2,
    1e-3,
    1e-2,
    1e-3,
    1e-2,
};

float32_t rms_prop_momentum_f[RMS_PROP_CASES] = {
    0.0,
    0.0,
    0.0,
    0.4,
    0.6,
    0.1,
};

float64_t rms_prop_momentum[RMS_PROP_CASES] = {
    0.0,
    0.0,
    0.0,
    0.4,
    0.6,
    0.1,
};

float32_t rms_prop_alpha_f[RMS_PROP_CASES] = {
    0.0,
    0.9,
    0.8,
    0.8,
    0.8,
    0.7,
};

float64_t rms_prop_alpha[RMS_PROP_CASES] = {
    0.0,
    0.9,
    0.8,
    0.8,
    0.8,
    0.7,
};

float32_t rms_prop_epsilon_f[RMS_PROP_CASES] = {
    1e-8,
    1e-7,
    1e-9,
    1e-9,
    1e-7,
    1e-9,
};

float64_t rms_prop_epsilon[RMS_PROP_CASES] = {
    1e-8,
    1e-7,
    1e-9,
    1e-9,
    1e-7,
    1e-9,
};

float32_t rms_prop_weight_decay_f[RMS_PROP_CASES] = {
    0.0,
    0.0,
    0.2,
    0.1,
    0.1,
    0.3,
};

float64_t rms_prop_weight_decay[RMS_PROP_CASES] = {
    0.0,
    0.0,
    0.2,
    0.1,
    0.1,
    0.3,
};

bool rms_prop_centered[RMS_PROP_CASES] = {
    false,
    false,
    false,
    false,
    true,
    true,
};

int rms_prop_iterations[RMS_PROP_CASES] = {
    5,
    5,
    5,
    5,
    5,
    5,
};

float32_t adam_learning_rate_f[ADAM_CASES] = {
    1e-3,
    1e-2,
    1e-1,
    1e-3,
    1e-2,
};

float64_t adam_learning_rate[ADAM_CASES] = {
    1e-3,
    1e-2,
    1e-1,
    1e-3,
    1e-2,
};

float32_t adam_beta_1_f[ADAM_CASES] = {
    0.9,
    0.9,
    0.8,
    0.7,
    0.0,
};

float64_t adam_beta_1[ADAM_CASES] = {
    0.9,
    0.9,
    0.8,
    0.7,
    0.0,
};

float32_t adam_beta_2_f[ADAM_CASES] = {
    0.995,
    0.995,
    0.9,
    0.7,
    0.0,
};

float64_t adam_beta_2[ADAM_CASES] = {
    0.995,
    0.995,
    0.9,
    0.7,
    0.0,
};

float32_t adam_weight_decay_f[ADAM_CASES] = {
    0.0,
    0.1,
    0.3,
    0.2,
    0.1,
};

float64_t adam_weight_decay[ADAM_CASES] = {
    0.0,
    0.1,
    0.3,
    0.2,
    0.1,
};

float32_t adam_epsilon_f[ADAM_CASES] = {
    1e-8,
    1e-7,
    1e-9,
    1e-7,
    1e-9,
};

float64_t adam_epsilon[ADAM_CASES] = {
    1e-8,
    1e-7,
    1e-9,
    1e-7,
    1e-9,
};

int adam_iterations[ADAM_CASES] = {
    5,
    5,
    5,
    5,
    5,
};

#define MODELS 3

typedef enum model_type_t
{
    CONVOLUTIONAL_NEURAL_NETWORK,
    SINGLE_LAYER_FEED_FORWARD,
    TRANSFORMER,
} model_type_t;

struct SingleLayerFeedForwardImpl : torch::nn::Module
{

    SingleLayerFeedForwardImpl() :
        hidden(register_module("hidden", torch::nn::Linear(5, 8))),
        output(register_module("output", torch::nn::Linear(8, 1))),
        hidden_activation(register_module("hidden_activation", torch::nn::ReLU())),
        output_activation(register_module("output_activation", torch::nn::Sigmoid()))
    {
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        return output_activation(output(hidden_activation(hidden(x))));
    }

    torch::nn::Linear hidden, output;
    torch::nn::ReLU hidden_activation;
    torch::nn::Sigmoid output_activation;
};
TORCH_MODULE(SingleLayerFeedForward);

struct ConvolutionalNeuralNetworkImpl : torch::nn::Module
{
    ConvolutionalNeuralNetworkImpl() :
        convtranspose1(register_module("convtranspose1", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(5, 2, 4).bias(false)))),
        convtranspose2(register_module("convtranspose2", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(2, 3, 3).stride(2).padding(1).bias(false)))),
        convtranspose3(register_module("convtranspose3", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(3, 4, 4).stride(2).padding(1).bias(false)))),
        convtranspose4(register_module("convtranspose4", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(4, 1, 3).stride(2).padding(1).bias(false)))),
        conv1(register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 2, 4).stride(2).padding(1).bias(false)))),
        conv2(register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(2, 3, 4).stride(2).padding(1).bias(false)))),
        conv3(register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 4, 4).stride(2).padding(1).bias(false)))),
        conv4(register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 1, 3).stride(1).padding(0).bias(false)))),
        batch_norm1(register_module("batch_norm1", torch::nn::BatchNorm2d(2))),
        batch_norm2(register_module("batch_norm2", torch::nn::BatchNorm2d(3))),
        batch_norm3(register_module("batch_norm3", torch::nn::BatchNorm2d(4))),
        batch_norm4(register_module("batch_norm4", torch::nn::BatchNorm2d(3))),
        batch_norm5(register_module("batch_norm5", torch::nn::BatchNorm2d(4))),
        leaky_relu1(register_module("leaky_relu1", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)))),
        leaky_relu2(register_module("leaky_relu2", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)))),
        leaky_relu3(register_module("leaky_relu3", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)))),
        relu1(register_module("relu1", torch::nn::ReLU())),
        relu2(register_module("relu2", torch::nn::ReLU())),
        relu3(register_module("relu3", torch::nn::ReLU())),
        tanh(register_module("tanh", torch::nn::Tanh())),
        sigmoid(register_module("sigmoid", torch::nn::Sigmoid()))
    {
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        x = relu1(batch_norm1(convtranspose1(x)));
        x = relu2(batch_norm2(convtranspose2(x)));
        x = relu3(batch_norm3(convtranspose3(x)));
        x = tanh(convtranspose4(x));
        x = leaky_relu1(conv1(x));
        x = leaky_relu2(batch_norm4(conv2(x)));
        x = leaky_relu3(batch_norm5(conv3(x)));
        x = sigmoid(conv4(x));
        return x.reshape({-1});
    }

    torch::nn::ConvTranspose2d convtranspose1, convtranspose2, convtranspose3, convtranspose4;
    torch::nn::Conv2d conv1, conv2, conv3, conv4;
    torch::nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3, batch_norm4, batch_norm5;
    torch::nn::LeakyReLU leaky_relu1, leaky_relu2, leaky_relu3;
    torch::nn::ReLU relu1, relu2, relu3;
    torch::nn::Tanh tanh;
    torch::nn::Sigmoid sigmoid;
};
TORCH_MODULE(ConvolutionalNeuralNetwork);

struct TransformerImpl : torch::nn::Module
{
    TransformerImpl(int64_t vocabulary_size, int64_t embedding_size, int64_t number_of_heads, int64_t block_size, float32_t epsilon) :
        linear1(register_module("linear1", torch::nn::Linear(torch::nn::LinearOptions(embedding_size, 4 * embedding_size)))), // embed X 4*embed
        linear2(register_module("linear2", torch::nn::Linear(torch::nn::LinearOptions(4 * embedding_size, embedding_size)))), // 4 * embed x embed
        linear3(register_module("linear3", torch::nn::Linear(torch::nn::LinearOptions(embedding_size, vocabulary_size)))), // embed X vocabulary_size
        layer_norm1(register_module("layer_norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_size}).elementwise_affine(true).eps(epsilon)))), // {embed}
        layer_norm2(register_module("layer_norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_size}).elementwise_affine(true).eps(epsilon)))),
        layer_norm3(register_module("layer_norm3", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_size}).elementwise_affine(true).eps(epsilon)))),
        dropout1(register_module("dropout1", torch::nn::Dropout(torch::nn::DropoutOptions(0.0)))),
        dropout2(register_module("dropout2", torch::nn::Dropout(torch::nn::DropoutOptions(0.0)))),
        token_embedding(register_module("token_embedding", torch::nn::Embedding(torch::nn::EmbeddingOptions(vocabulary_size, embedding_size)))), // vocab X embed
        position_embedding(register_module("position_embedding", torch::nn::Embedding(torch::nn::EmbeddingOptions(block_size, embedding_size)))), // block_size X embed
        multihead_attention(register_module("multihead_attention", torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(embedding_size, number_of_heads)
                                                                                                .dropout(0.0).add_bias_kv(false).add_zero_attn(false).bias(false)))),
        relu(register_module("relu", torch::nn::ReLU()))
    {
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        torch::Tensor mask = torch::zeros({x.size(1), x.size(1)}).where(torch::ones({x.size(1), x.size(1)}, torch::TensorOptions().dtype(torch::kBool)).triu(1) == 0, -std::numeric_limits<float>::infinity());
        torch::Tensor positions = torch::arange(x.size(1), torch::TensorOptions().dtype(torch::kLong));
        x = dropout1(token_embedding(x.to(torch::kLong)) + position_embedding(positions));
        torch::Tensor y = layer_norm1(x).transpose(0, 1);
        x = x + std::get<0>(multihead_attention->forward(y, y, y, {}, false, mask, false)).transpose(0, 1);
        x = x + dropout2(linear2(relu(linear1(layer_norm2(x)))));
        x = linear3(layer_norm3(x));
        return x.reshape({-1, x.size(-1)});
    }

    torch::nn::Linear linear1, linear2, linear3;
    torch::nn::LayerNorm layer_norm1, layer_norm2, layer_norm3;
    torch::nn::Dropout dropout1, dropout2;
    torch::nn::Embedding token_embedding, position_embedding;
    torch::nn::MultiheadAttention multihead_attention;
    torch::nn::ReLU relu;
};
TORCH_MODULE(Transformer);

nw_error_t *error = NULL;
std::vector<optimizer_t *> optimizers[RUNTIMES][DATATYPES][MODELS];
std::vector<torch::optim::SGD> torch_optimizers_sgd[RUNTIMES][DATATYPES][MODELS];
std::vector<torch::optim::RMSprop> torch_optimizers_rms_prop[RUNTIMES][DATATYPES][MODELS];
std::vector<torch::optim::Adam> torch_optimizers_adam[RUNTIMES][DATATYPES][MODELS];
std::vector<model_t *> models[RUNTIMES][DATATYPES][MODELS];
std::vector<SingleLayerFeedForward> torch_models_single_layer_feed_forward[RUNTIMES][DATATYPES];
std::vector<ConvolutionalNeuralNetwork> torch_models_convolutional_neural_network[RUNTIMES][DATATYPES];
std::vector<Transformer> torch_models_transformer[RUNTIMES][DATATYPES];
std::vector<tensor_t *> inputs[RUNTIMES][DATATYPES][MODELS];
std::vector<tensor_t *> outputs[RUNTIMES][DATATYPES][MODELS];
std::vector<torch::Tensor> torch_inputs[RUNTIMES][DATATYPES][MODELS];
std::vector<torch::Tensor> torch_outputs[RUNTIMES][DATATYPES][MODELS];

int cases(algorithm_type_t algorithm_type)
{
    switch(algorithm_type)
    {
    case STOCASTIC_GRADIENT_DESCENT:
        return STOCASTIC_GRADIENT_DESCENT_CASES;
    case RMS_PROP:
        return RMS_PROP_CASES;
    case ADAM:
        return ADAM_CASES;
    default:
        return 0;
    }
}

int iterations(algorithm_type_t algorithm_type, int test_case)
{
    switch(algorithm_type)
    {
    case STOCASTIC_GRADIENT_DESCENT:
        return sgd_iterations[test_case];
    case RMS_PROP:
        return rms_prop_iterations[test_case];
    case ADAM:
        return adam_iterations[test_case];
    default:
        return 0;
    }
}

void setup_single_layer_feed_forward(runtime_t runtime, datatype_t datatype)
{
    // Torch Model
    torch::Tensor torch_input;
    torch::Tensor torch_output;
    switch (datatype)
    {
    case FLOAT32:
        torch_input = torch::randn({8, 5}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
        torch_output = torch::randn({8, 1}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
        torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(torch::kFloat32));
        break;
    case FLOAT64:
        torch_input = torch::randn({8, 5}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(false));
        torch_output = torch::randn({8, 1}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(false));
        torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(torch::kFloat64));
        break;
    default:
        ck_abort_msg("unknown data type.");
    }
    SingleLayerFeedForward single_layer_feed_forward = SingleLayerFeedForward();
    torch_models_single_layer_feed_forward[runtime][datatype].push_back(single_layer_feed_forward);
    torch_inputs[runtime][datatype][SINGLE_LAYER_FEED_FORWARD].push_back(torch_input);
    torch_outputs[runtime][datatype][SINGLE_LAYER_FEED_FORWARD].push_back(torch_output);

    // NW Model
    model_t *model = NULL;
    tensor_t *input = torch_to_tensor(torch_input, runtime, datatype);
    tensor_t *output = torch_to_tensor(torch_output, runtime, datatype);

    layer_t *hidden_layer = NULL;
    layer_t *output_layer = NULL;
    layer_t *hidden_activation_layer = NULL;
    layer_t *output_activation_layer = NULL;
    block_t *block = NULL;

    error = rectified_linear_activation_layer_create(&hidden_activation_layer);
    ck_assert_ptr_null(error);
    ck_assert_ptr_nonnull(hidden_activation_layer);

    error = sigmoid_activation_layer_create(&output_activation_layer);
    ck_assert_ptr_null(error);
    ck_assert_ptr_nonnull(output_activation_layer);

    error = linear_layer_create_from_parameters(&hidden_layer, torch_to_tensor(single_layer_feed_forward->hidden->weight.t(), runtime, datatype), 
                                                torch_to_tensor(single_layer_feed_forward->hidden->bias, runtime, datatype));
    ck_assert_ptr_null(error);
    ck_assert_ptr_nonnull(hidden_layer);

    error = linear_layer_create_from_parameters(&output_layer, torch_to_tensor(single_layer_feed_forward->output->weight.t(), runtime, datatype), 
                                                torch_to_tensor(single_layer_feed_forward->output->bias, runtime, datatype));
    ck_assert_ptr_null(error);
    ck_assert_ptr_nonnull(output_layer);

    error = block_create(&block, 4, hidden_layer, hidden_activation_layer, output_layer, output_activation_layer);
    ck_assert_ptr_null(error);
    ck_assert_ptr_nonnull(block);

    error = model_create(&model, block);
    ck_assert_ptr_null(error);

    models[runtime][datatype][SINGLE_LAYER_FEED_FORWARD].push_back(model);
    inputs[runtime][datatype][SINGLE_LAYER_FEED_FORWARD].push_back(input);
    outputs[runtime][datatype][SINGLE_LAYER_FEED_FORWARD].push_back(output);
}

void setup_convolutional_neural_network(runtime_t runtime, datatype_t datatype)
{
        torch::Tensor torch_input;
        torch::Tensor torch_output;
        switch (datatype)
        {
        case FLOAT32:
            torch_input = torch::rand({3, 5, 1, 1}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
            torch_output = torch::rand({3}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
            torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(torch::kFloat32));
            break;
        case FLOAT64:
            torch_input = torch::rand({3, 5, 1, 1}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(false));
            torch_output = torch::rand({3}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(false));
            torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(torch::kFloat64));
            break;
        default:
            ck_abort_msg("unknown data type.");
        }
        ConvolutionalNeuralNetwork convolutional_neural_network = ConvolutionalNeuralNetwork();
        torch_models_convolutional_neural_network[runtime][datatype].push_back(convolutional_neural_network);
        torch_inputs[runtime][datatype][CONVOLUTIONAL_NEURAL_NETWORK].push_back(torch_input);
        torch_outputs[runtime][datatype][CONVOLUTIONAL_NEURAL_NETWORK].push_back(torch_output);

        // NW Model
        model_t *model = NULL;
        tensor_t *input = torch_to_tensor(torch_input, runtime, datatype);
        tensor_t *output = torch_to_tensor(torch_output, runtime, datatype);

        layer_t *convtranspose1 = NULL, *convtranspose2 = NULL, *convtranspose3 = NULL, *convtranspose4 = NULL;
        layer_t *relu1 = NULL, *relu2 = NULL, *relu3 = NULL;
        layer_t *conv1 = NULL, *conv2 = NULL, *conv3 = NULL, *conv4 = NULL;
        layer_t *batch_norm1 = NULL, *batch_norm2 = NULL, *batch_norm3 = NULL, *batch_norm4 = NULL, *batch_norm5 = NULL;
        layer_t *leaky_relu1 = NULL, *leaky_relu2 = NULL, *leaky_relu3 = NULL;
        void *momentum = NULL, *epsilon = NULL, *c = NULL;
        layer_t *tanh = NULL;
        layer_t *sigmoid = NULL;
        layer_t *reshape = NULL;
        block_t *block = NULL;

        momentum = (void *) malloc(datatype_size(datatype));
        ck_assert_ptr_nonnull(momentum);
        epsilon = (void *) malloc(datatype_size(datatype));
        ck_assert_ptr_nonnull(epsilon);
        c = (void *) malloc(datatype_size(datatype));
        ck_assert_ptr_nonnull(c);

        switch (datatype)
        {
        case FLOAT32:
            *(float32_t *) momentum = (float32_t) 0.1;
            *(float32_t *) epsilon = (float32_t) 1e-5;
            *(float32_t *) c = (float32_t) 0.2;
            break;
        case FLOAT64:
            *(float64_t *) momentum = (float64_t) 0.1;
            *(float64_t *) epsilon = (float64_t) 1e-5;
            *(float64_t *) c = (float64_t) 0.2;
            break;
        default:
            ck_abort_msg("unknown datatype");
        }

        error = convolution_transpose_2d_layer_create_from_parameters(&convtranspose1, 0, 1, torch_to_tensor(convolutional_neural_network->convtranspose1->weight, runtime, datatype), NULL);
        ck_assert_ptr_null(error);

        error = convolution_transpose_2d_layer_create_from_parameters(&convtranspose2, 1, 2, torch_to_tensor(convolutional_neural_network->convtranspose2->weight, runtime, datatype), NULL);
        ck_assert_ptr_null(error);

        error = convolution_transpose_2d_layer_create_from_parameters(&convtranspose3, 1, 2, torch_to_tensor(convolutional_neural_network->convtranspose3->weight, runtime, datatype), NULL);
        ck_assert_ptr_null(error);

        error = convolution_transpose_2d_layer_create_from_parameters(&convtranspose4, 1, 2, torch_to_tensor(convolutional_neural_network->convtranspose4->weight, runtime, datatype), NULL);
        ck_assert_ptr_null(error);

        error = convolution_2d_layer_create_from_parameters(&conv1, 1, 2, torch_to_tensor(convolutional_neural_network->conv1->weight, runtime, datatype), NULL);
        ck_assert_ptr_null(error);

        error = convolution_2d_layer_create_from_parameters(&conv2, 1, 2, torch_to_tensor(convolutional_neural_network->conv2->weight, runtime, datatype), NULL);
        ck_assert_ptr_null(error);

        error = convolution_2d_layer_create_from_parameters(&conv3, 1, 2, torch_to_tensor(convolutional_neural_network->conv3->weight, runtime, datatype), NULL);
        ck_assert_ptr_null(error);

        error = convolution_2d_layer_create_from_parameters(&conv4, 0, 1, torch_to_tensor(convolutional_neural_network->conv4->weight, runtime, datatype), NULL);
        ck_assert_ptr_null(error);

        error = batch_normalization_2d_layer_create(&batch_norm1, 2, momentum, epsilon, true, true, datatype, runtime);
        ck_assert_ptr_null(error);

        error = batch_normalization_2d_layer_create(&batch_norm2, 3, momentum, epsilon, true, true, datatype, runtime);
        ck_assert_ptr_null(error);

        error = batch_normalization_2d_layer_create(&batch_norm3, 4, momentum, epsilon, true, true, datatype, runtime);
        ck_assert_ptr_null(error);

        error = batch_normalization_2d_layer_create(&batch_norm4, 3, momentum, epsilon, true, true, datatype, runtime);
        ck_assert_ptr_null(error);

        error = batch_normalization_2d_layer_create(&batch_norm5, 4, momentum, epsilon, true, true, datatype, runtime);
        ck_assert_ptr_null(error);

        error = rectified_linear_activation_layer_create(&relu1);
        ck_assert_ptr_null(error);

        error = rectified_linear_activation_layer_create(&relu2);
        ck_assert_ptr_null(error);

        error = rectified_linear_activation_layer_create(&relu3);
        ck_assert_ptr_null(error);

        error = leaky_rectified_linear_activation_layer_create(&leaky_relu1, c, datatype);
        ck_assert_ptr_null(error);

        error = leaky_rectified_linear_activation_layer_create(&leaky_relu2, c, datatype);
        ck_assert_ptr_null(error);

        error = leaky_rectified_linear_activation_layer_create(&leaky_relu3, c, datatype);
        ck_assert_ptr_null(error);

        error = tanh_activation_layer_create(&tanh);
        ck_assert_ptr_null(error);

        error = sigmoid_activation_layer_create(&sigmoid);
        ck_assert_ptr_null(error);

        int64_t shape[] = {3};
        error = reshape_layer_create(&reshape, shape, 1);
        ck_assert_ptr_null(error);

        error = block_create(&block, 22, convtranspose1, batch_norm1, relu1, convtranspose2, batch_norm2, relu2, convtranspose3, batch_norm3, relu3, convtranspose4,
                             tanh, conv1, leaky_relu1, conv2, batch_norm4, leaky_relu2, conv3, batch_norm5, leaky_relu3, conv4, sigmoid, reshape);
        ck_assert_ptr_null(error);

        error = model_create(&model, block);
        ck_assert_ptr_null(error);

        models[runtime][datatype][CONVOLUTIONAL_NEURAL_NETWORK].push_back(model);
        inputs[runtime][datatype][CONVOLUTIONAL_NEURAL_NETWORK].push_back(input);
        outputs[runtime][datatype][CONVOLUTIONAL_NEURAL_NETWORK].push_back(output);

        free(momentum);
        free(epsilon);
        free(c);
}

void setup_transformer(runtime_t runtime, datatype_t datatype)
{
    // Paramters
    int64_t batch_size = 2;
    int64_t vocabulary_size = 5;
    int64_t embedding_size = 4;
    int64_t number_of_heads = 2;
    int64_t block_size = 3;
    void *epsilon = NULL;
    void *dropout_probability = NULL;

    epsilon = (void *) malloc(datatype_size(datatype));
    ck_assert_ptr_nonnull(epsilon);
    dropout_probability = (void *) malloc(datatype_size(datatype));
    ck_assert_ptr_nonnull(dropout_probability);

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) epsilon = (float32_t) 1e-5;
        *(float32_t *) dropout_probability = (float32_t) 0.0;
        break;
    case FLOAT64:
        *(float64_t *) epsilon = (float64_t) 1e-5;
        *(float64_t *) dropout_probability = (float64_t) 0.0;
        break;
    default:
        ck_abort_msg("unknown datatype");
    }

    // Torch Model
    torch::Tensor torch_input;
    torch::Tensor torch_output;
    switch (datatype)
    {
    case FLOAT32:
        torch_input = torch::randint(0, vocabulary_size, {batch_size, block_size}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
        torch_output = torch::randint(0, vocabulary_size, {batch_size * block_size, 1}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
        torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(torch::kFloat32));
        break;
    case FLOAT64:
        torch_input = torch::randint(0, vocabulary_size, {batch_size, block_size}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(false));
        torch_output = torch::randint(0, vocabulary_size, {batch_size * block_size, 1}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(false));
        torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(torch::kFloat64));
        break;
    default:
        ck_abort_msg("unknown data type.");
    }
    Transformer transformer = Transformer(vocabulary_size, embedding_size, number_of_heads, block_size, 1e-5);
    torch_models_transformer[runtime][datatype].push_back(transformer);
    torch_inputs[runtime][datatype][TRANSFORMER].push_back(torch_input);
    torch_outputs[runtime][datatype][TRANSFORMER].push_back(torch_output);

    // NW Model
    model_t *model = NULL;
    tensor_t *input = torch_to_tensor(torch_input, runtime, datatype);
    tensor_t *output = torch_to_tensor(torch_output, runtime, datatype);

    int64_t normal_shape[] = {embedding_size};
    // int64_t output_shape[] = {batch_size * block_size, vocabulary_size};
    int64_t output_shape[] = {-1, vocabulary_size};
    layer_t *layer_norm_1 = NULL, *layer_norm_2 = NULL, *layer_norm_3;
    layer_t *causal_multihead_self_attention = NULL;
    layer_t *linear_1 = NULL, *linear_2 = NULL, *linear_3 = NULL;
    layer_t *relu = NULL;
    layer_t *dropout_1 = NULL, *dropout_2 = NULL;
    block_t *residual_block_1 = NULL;
    block_t *residual_block_2 = NULL;
    layer_t *residual_block_layer_1 = NULL;
    layer_t *residual_block_layer_2 = NULL;
    block_t *transformer_block = NULL;
    layer_t *transformer_embedding = NULL;
    layer_t *decoder = NULL;
    block_t *block = NULL;
    layer_t *reshape = NULL;

    error = reshape_layer_create(&reshape, output_shape, 2);
    ck_assert_ptr_null(error);

    error = layer_normalization_layer_create(&layer_norm_1, normal_shape, 1, epsilon, true, datatype, runtime);
    ck_assert_ptr_null(error);

    error = layer_normalization_layer_create(&layer_norm_2, normal_shape, 1, epsilon, true, datatype, runtime);
    ck_assert_ptr_null(error);

    error = layer_normalization_layer_create(&layer_norm_3, normal_shape, 1, epsilon, true, datatype, runtime);
    ck_assert_ptr_null(error);

    error = causal_multihead_self_attention_layer_create_from_parameters(&causal_multihead_self_attention, number_of_heads, embedding_size, dropout_probability, datatype,
                                                                         torch_to_tensor(transformer->multihead_attention->in_proj_weight.t(), runtime, datatype),
                                                                         NULL,
                                                                         torch_to_tensor(transformer->multihead_attention->out_proj->weight.t(), runtime, datatype),
                                                                         NULL);
    ck_assert_ptr_null(error);

    error = linear_layer_create_from_parameters(&linear_1, torch_to_tensor(transformer->linear1->weight.t(), runtime, datatype), torch_to_tensor(transformer->linear1->bias, runtime, datatype));
    ck_assert_ptr_null(error);

    error = linear_layer_create_from_parameters(&linear_2, torch_to_tensor(transformer->linear2->weight.t(), runtime, datatype), torch_to_tensor(transformer->linear2->bias, runtime, datatype));
    ck_assert_ptr_null(error);

    error = linear_layer_create_from_parameters(&linear_3, torch_to_tensor(transformer->linear3->weight.t(), runtime, datatype), torch_to_tensor(transformer->linear3->bias, runtime, datatype));
    ck_assert_ptr_null(error);

    error = dropout_layer_create(&dropout_1, dropout_probability, datatype);
    ck_assert_ptr_null(error);

    error = dropout_layer_create(&dropout_2, dropout_probability, datatype);
    ck_assert_ptr_null(error);

    error = rectified_linear_activation_layer_create(&relu);
    ck_assert_ptr_null(error);

    error = block_create(&residual_block_1, 2, layer_norm_1, causal_multihead_self_attention);
    ck_assert_ptr_null(error);

    error = block_create(&residual_block_2, 5, layer_norm_2, linear_1, relu, linear_2, dropout_2);
    ck_assert_ptr_null(error);

    error = residual_block_layer_create(&residual_block_layer_1, residual_block_1);
    ck_assert_ptr_null(error);

    error = residual_block_layer_create(&residual_block_layer_2, residual_block_2);
    ck_assert_ptr_null(error);

    error = block_create(&transformer_block, 2, residual_block_layer_1, residual_block_layer_2);
    ck_assert_ptr_null(error);

    error = block_layer_create(&decoder, transformer_block);
    ck_assert_ptr_null(error);

    error = transformer_embedding_layer_create_from_parameters(&transformer_embedding, torch_to_tensor(transformer->token_embedding->weight, runtime, datatype), 
                                                               torch_to_tensor(transformer->position_embedding->weight, runtime, datatype));
    ck_assert_ptr_null(error);

    error = block_create(&block, 6, transformer_embedding, dropout_1, decoder, layer_norm_3, linear_3, reshape);
    ck_assert_ptr_null(error);

    error = model_create(&model, block);
    ck_assert_ptr_null(error);

    models[runtime][datatype][TRANSFORMER].push_back(model);
    inputs[runtime][datatype][TRANSFORMER].push_back(input);
    outputs[runtime][datatype][TRANSFORMER].push_back(output);

    free(epsilon);
    free(dropout_probability);
}

void setup_model(runtime_t runtime, datatype_t datatype, model_type_t model_type)
{
    switch (model_type)
    {
    case SINGLE_LAYER_FEED_FORWARD:
        setup_single_layer_feed_forward(runtime, datatype);
        break;
    case CONVOLUTIONAL_NEURAL_NETWORK:
        setup_convolutional_neural_network(runtime, datatype);
        break;
    case TRANSFORMER:
        setup_transformer(runtime, datatype);
        break;
    default:
        ck_abort_msg("unknown model");
    }
}

void setup_optimizer(algorithm_type_t algorithm_type)
{
    const int CASES = cases(algorithm_type);
    for (int i = 0; i < RUNTIMES; ++i)
    {
        runtime_t runtime = (runtime_t) i;
        runtime_create_context(runtime);
        for (int j = 0; j < DATATYPES; ++j)
        {
            datatype_t datatype = (datatype_t) j;
            for (int k = 0; k < MODELS; ++k)
            {
                model_type_t model_type = (model_type_t) k;
                optimizers[i][j][k].clear();
                optimizers[i][j][k] = std::vector<optimizer_t *>(CASES);
                inputs[i][j][k].clear();
                outputs[i][j][k].clear();
                models[i][j][k].clear();
                torch_inputs[i][j][k].clear();
                torch_outputs[i][j][k].clear();
                switch (model_type)
                {
                case SINGLE_LAYER_FEED_FORWARD:
                    torch_models_single_layer_feed_forward[i][j].clear();
                    break;
                case CONVOLUTIONAL_NEURAL_NETWORK:
                    torch_models_convolutional_neural_network[i][j].clear();
                    break;
                case TRANSFORMER:
                    torch_models_transformer[i][j].clear();
                    break;
                default:
                    ck_abort_msg("unknwown model type.");
                }
                switch (algorithm_type)
                {
                case STOCASTIC_GRADIENT_DESCENT:
                    torch_optimizers_sgd[i][j][k].clear();
                    break;
                case RMS_PROP:
                    torch_optimizers_rms_prop[i][j][k].clear();
                    break;
                case ADAM:
                    torch_optimizers_adam[i][j][k].clear();
                    break;
                default:
                    ck_abort_msg("unknown optimizer.");
                }
                for (int l = 0; l < CASES; ++l)
                {
                    setup_model(runtime, datatype, model_type);
                    torch::autograd::variable_list parameters;

                    switch (model_type)
                    {
                    case SINGLE_LAYER_FEED_FORWARD:
                        parameters = torch_models_single_layer_feed_forward[i][j][l]->parameters();
                        break;
                    case CONVOLUTIONAL_NEURAL_NETWORK:
                        parameters = torch_models_convolutional_neural_network[i][j][l]->parameters();
                        break;
                    case TRANSFORMER:
                        parameters = torch_models_transformer[i][j][l]->parameters();
                        break;
                    default:
                        ck_abort_msg("unknown model.");
                    }

                    switch (algorithm_type)
                    {
                    case STOCASTIC_GRADIENT_DESCENT:
                        switch (datatype)
                        {
                        case FLOAT32:
                            error = optimizer_stochastic_gradient_descent_create(&optimizers[i][j][k][l], datatype, (void *) &sgd_learning_rate_f[l], 
                                                                                 (void *) &sgd_momentum_f[l], (void *) &sgd_dampening_f[l], 
                                                                                 (void *) &sgd_weight_decay_f[l], sgd_nesterov[l]);
                            torch_optimizers_sgd[i][j][k].push_back(torch::optim::SGD(parameters, torch::optim::SGDOptions(sgd_learning_rate_f[l])
                                        .momentum(sgd_momentum_f[l]).dampening(sgd_dampening_f[l]).weight_decay(sgd_weight_decay_f[l]).nesterov(sgd_nesterov[l])));
                            break;
                        case FLOAT64:
                            error = optimizer_stochastic_gradient_descent_create(&optimizers[i][j][k][l], datatype, (void *) &sgd_learning_rate[l], 
                                                                                 (void *) &sgd_momentum[l], (void *) &sgd_dampening[l], 
                                                                                 (void *) &sgd_weight_decay[l], sgd_nesterov[l]);
                            torch_optimizers_sgd[i][j][k].push_back(torch::optim::SGD(parameters, torch::optim::SGDOptions(sgd_learning_rate[l])
                                        .momentum(sgd_momentum[l]).dampening(sgd_dampening[l]).weight_decay(sgd_weight_decay[l]).nesterov(sgd_nesterov[l])));
                            break;
                        default:
                            ck_abort_msg("unknown data type.");
                        }
                        break;
                    case RMS_PROP:
                        switch (datatype)
                        {
                        case FLOAT32:
                            error = optimizer_rms_prop_create(&optimizers[i][j][k][l], datatype, (void *) &rms_prop_learning_rate_f[l], 
                                                             (void *) &rms_prop_momentum_f[l], (void *) &rms_prop_alpha_f[l], 
                                                             (void *) &rms_prop_weight_decay_f[l], (void *) &rms_prop_epsilon_f[l], rms_prop_centered[l]);
                            torch_optimizers_rms_prop[i][j][k].push_back(torch::optim::RMSprop(parameters, torch::optim::RMSpropOptions(rms_prop_learning_rate_f[l])
                                .eps(rms_prop_epsilon_f[l]).momentum(rms_prop_momentum_f[l]).alpha(rms_prop_alpha_f[l]).weight_decay(rms_prop_weight_decay_f[l])
                                .centered(rms_prop_centered[l])));
                            break;
                        case FLOAT64:
                            error = optimizer_rms_prop_create(&optimizers[i][j][k][l], datatype, (void *) &rms_prop_learning_rate[l], 
                                                             (void *) &rms_prop_momentum[l], (void *) &rms_prop_alpha[l], 
                                                             (void *) &rms_prop_weight_decay[l], (void *) &rms_prop_epsilon[l], rms_prop_centered[l]);
                            torch_optimizers_rms_prop[i][j][k].push_back(torch::optim::RMSprop(parameters, torch::optim::RMSpropOptions(rms_prop_learning_rate[l])
                                .eps(rms_prop_epsilon[l]).momentum(rms_prop_momentum[l]).alpha(rms_prop_alpha[l]).weight_decay(rms_prop_weight_decay[l])
                                .centered(rms_prop_centered[l])));
                            break;
                        default:
                            ck_abort_msg("unknown data type.");
                        }
                        break;
                    case ADAM:
                        switch (datatype)
                        {
                        case FLOAT32:
                            error = optimizer_adam_create(&optimizers[i][j][k][l], datatype, (void *) &adam_learning_rate_f[l], 
                                                          (void *) &adam_beta_1_f[l], (void *) &adam_beta_2_f[l], 
                                                          (void *) &adam_weight_decay_f[l], (void *) &adam_epsilon_f[l]);
                            torch_optimizers_adam[i][j][k].push_back(torch::optim::Adam(parameters, torch::optim::AdamOptions(adam_learning_rate_f[l])
                                        .betas(std::make_tuple(adam_beta_1_f[l], adam_beta_2_f[l])).eps(adam_epsilon_f[l]).weight_decay(adam_weight_decay_f[l])));
                            break;
                        case FLOAT64:
                            error = optimizer_adam_create(&optimizers[i][j][k][l], datatype, (void *) &adam_learning_rate[l], 
                                                          (void *) &adam_beta_1[l], (void *) &adam_beta_2[l], 
                                                          (void *) &adam_weight_decay[l], (void *) &adam_epsilon[l]);
                            torch_optimizers_adam[i][j][k].push_back(torch::optim::Adam(parameters, torch::optim::AdamOptions(adam_learning_rate[l])
                                        .betas(std::make_tuple(adam_beta_1[l], adam_beta_2[l])).eps(adam_epsilon[l]).weight_decay(adam_weight_decay[l])));
                            break;
                        default:
                            ck_abort_msg("unknown data type.");
                        }
                        break;
                    default:
                        ck_abort_msg("unknown optimization algorithm.");
                    }
                }
            }
        }
    }
}

void teardown_optimizer(algorithm_type_t algorithm_type)
{
    error_print(error);
    error_destroy(error);
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < MODELS; ++k)
            {
                for (int l = 0; l < cases(algorithm_type); ++l)
                {
                    model_destroy(models[i][j][k][l]);
                    optimizer_destroy(optimizers[i][j][k][l]);
                    tensor_destroy(inputs[i][j][k][l]);
                    tensor_destroy(outputs[i][j][k][l]);

                    models[i][j][k][l] = NULL;
                    optimizers[i][j][k][l] = NULL;
                    inputs[i][j][k][l] = NULL;
                    outputs[i][j][k][l] = NULL;
                }
            }
        }
    }
}

void ck_compare_linear(torch::nn::Linear torch_linear, linear_t *linear, runtime_t runtime, datatype_t datatype)
{
    tensor_t *torch_weights = torch_to_tensor(torch_linear->weight.t(), runtime, datatype); 
    tensor_t *torch_bias = (linear->bias) ? torch_to_tensor(torch_linear->bias, runtime, datatype) : NULL;

    ck_assert_tensor_equiv(linear->weights, torch_weights);
    if (linear->bias)
    {
        ck_assert_tensor_equiv(linear->bias, torch_bias);
    }

    tensor_destroy(torch_weights);
    tensor_destroy(torch_bias);
}

void ck_compare_convolution_transpose_2d(torch::nn::ConvTranspose2d torch_conv2d, convolution_2d_t *convolution_2d, runtime_t runtime, datatype_t datatype)
{
    tensor_t *torch_weights = torch_to_tensor(torch_conv2d->weight, runtime, datatype); 
    tensor_t *torch_bias = (convolution_2d->bias) ? torch_to_tensor(torch_conv2d->bias, runtime, datatype) : NULL;

    ck_assert_tensor_equiv(convolution_2d->kernel, torch_weights);
    if (convolution_2d->bias)
    {
        ck_assert_tensor_equiv(convolution_2d->bias, torch_bias);
    }

    tensor_destroy(torch_weights);
    tensor_destroy(torch_bias);
}

void ck_compare_convolution_2d(torch::nn::Conv2d torch_conv2d, convolution_2d_t *convolution_2d, runtime_t runtime, datatype_t datatype)
{
    tensor_t *torch_weights = torch_to_tensor(torch_conv2d->weight, runtime, datatype); 
    tensor_t *torch_bias = (convolution_2d->bias) ? torch_to_tensor(torch_conv2d->bias, runtime, datatype) : NULL;

    ck_assert_tensor_equiv(convolution_2d->kernel, torch_weights);
    if (convolution_2d->bias)
    {
        ck_assert_tensor_equiv(convolution_2d->bias, torch_bias);
    }

    tensor_destroy(torch_weights);
    tensor_destroy(torch_bias);
}

void ck_compare_batch_normalization_2d(torch::nn::BatchNorm2d torch_batch_normalization_2d, batch_normalization_2d_t *batch_normalization_2d, runtime_t runtime, datatype_t datatype)
{
    tensor_t *torch_weights = (batch_normalization_2d->weights) ? torch_to_tensor(torch_batch_normalization_2d->weight, runtime, datatype) : NULL;
    tensor_t *torch_bias = (batch_normalization_2d->bias) ? torch_to_tensor(torch_batch_normalization_2d->bias, runtime, datatype) : NULL;
    tensor_t *torch_running_mean = torch_to_tensor(torch_batch_normalization_2d->running_mean, runtime, datatype);
    tensor_t *torch_running_variance = torch_to_tensor(torch_batch_normalization_2d->running_var, runtime, datatype);

    if (batch_normalization_2d->weights)
    {
        ck_assert_tensor_equiv(batch_normalization_2d->weights, torch_weights);
    }
    if (batch_normalization_2d->bias)
    {
        ck_assert_tensor_equiv(batch_normalization_2d->bias, torch_bias);
    }
    ck_assert_tensor_equiv(batch_normalization_2d->running_mean, torch_running_mean);
    ck_assert_tensor_equiv(batch_normalization_2d->running_variance, torch_running_variance);

    tensor_destroy(torch_weights);
    tensor_destroy(torch_bias);
    tensor_destroy(torch_running_mean);
    tensor_destroy(torch_running_variance);
}

void ck_compare_layer_normalization(torch::nn::LayerNorm torch_layer_normalization, layer_normalization_t *layer_normalization, runtime_t runtime, datatype_t datatype)
{
    tensor_t *torch_weights = (layer_normalization->weights) ? torch_to_tensor(torch_layer_normalization->weight, runtime, datatype) : NULL;
    tensor_t *torch_bias = (layer_normalization->bias) ? torch_to_tensor(torch_layer_normalization->bias, runtime, datatype) : NULL;
    if (layer_normalization->weights)
    {
        ck_assert_tensor_equiv(layer_normalization->weights, torch_weights);
    }
    if (layer_normalization->bias)
    {
        ck_assert_tensor_equiv(layer_normalization->bias, torch_bias);
    }
    tensor_destroy(torch_weights);
    tensor_destroy(torch_bias);
}

void ck_compare_embedding(torch::nn::Embedding torch_embedding, embedding_t *embedding, runtime_t runtime, datatype_t datatype)
{
    tensor_t *torch_weights = torch_to_tensor(torch_embedding->weight, runtime, datatype);
    ck_assert_tensor_equiv(embedding->weights, torch_weights);
    tensor_destroy(torch_weights);
}

void ck_compare_multihead_attention(torch::nn::MultiheadAttention torch_multihead_attention, causal_multihead_self_attention_t *multihead_attention, runtime_t runtime, datatype_t datatype)
{
    tensor_t *torch_input_weights = torch_to_tensor(torch_multihead_attention->in_proj_weight.t(), runtime, datatype);
    tensor_t *torch_input_bias = (multihead_attention->input_bias) ? torch_to_tensor(torch_multihead_attention->in_proj_bias, runtime, datatype) : NULL;
    tensor_t *torch_output_weights = torch_to_tensor(torch_multihead_attention->out_proj->weight.t(), runtime, datatype);
    tensor_t *torch_output_bias = (multihead_attention->output_bias) ? torch_to_tensor(torch_multihead_attention->out_proj->bias, runtime, datatype) : NULL;

    ck_assert_tensor_equiv(multihead_attention->input_weights, torch_input_weights);
    if (multihead_attention->input_bias)
    {
        ck_assert_tensor_equiv(multihead_attention->input_bias, torch_input_bias);
    }

    ck_assert_tensor_equiv(multihead_attention->output_weights, torch_output_weights);
    if (multihead_attention->output_bias)
    {
        ck_assert_tensor_equiv(multihead_attention->output_bias, torch_output_bias);
    }

    tensor_destroy(torch_input_weights);
    tensor_destroy(torch_input_bias);
    tensor_destroy(torch_output_weights);
    tensor_destroy(torch_output_bias);
}

void ck_compare_models(runtime_t runtime, datatype_t datatype, model_type_t model_type, int test_case)
{

    switch (model_type)
    {
    case SINGLE_LAYER_FEED_FORWARD:
        ck_compare_linear(torch_models_single_layer_feed_forward[runtime][datatype][test_case]->hidden,
                          models[runtime][datatype][model_type][test_case]->block->layers[0]->transform->linear, runtime, datatype);
        ck_compare_linear(torch_models_single_layer_feed_forward[runtime][datatype][test_case]->output,
                          models[runtime][datatype][model_type][test_case]->block->layers[2]->transform->linear, runtime, datatype);
        break;
    case CONVOLUTIONAL_NEURAL_NETWORK:
        ck_compare_convolution_transpose_2d(torch_models_convolutional_neural_network[runtime][datatype][test_case]->convtranspose1,
                                            models[runtime][datatype][model_type][test_case]->block->layers[0]->transform->convolution_2d, runtime, datatype);
        ck_compare_convolution_transpose_2d(torch_models_convolutional_neural_network[runtime][datatype][test_case]->convtranspose2,
                                            models[runtime][datatype][model_type][test_case]->block->layers[3]->transform->convolution_2d, runtime, datatype);
        ck_compare_convolution_transpose_2d(torch_models_convolutional_neural_network[runtime][datatype][test_case]->convtranspose3,
                                            models[runtime][datatype][model_type][test_case]->block->layers[6]->transform->convolution_2d, runtime, datatype);
        ck_compare_convolution_transpose_2d(torch_models_convolutional_neural_network[runtime][datatype][test_case]->convtranspose4,
                                            models[runtime][datatype][model_type][test_case]->block->layers[9]->transform->convolution_2d, runtime, datatype);
        ck_compare_convolution_2d(torch_models_convolutional_neural_network[runtime][datatype][test_case]->conv1,
                                  models[runtime][datatype][model_type][test_case]->block->layers[11]->transform->convolution_2d, runtime, datatype);
        ck_compare_convolution_2d(torch_models_convolutional_neural_network[runtime][datatype][test_case]->conv2,
                                  models[runtime][datatype][model_type][test_case]->block->layers[13]->transform->convolution_2d, runtime, datatype);
        ck_compare_convolution_2d(torch_models_convolutional_neural_network[runtime][datatype][test_case]->conv3,
                                  models[runtime][datatype][model_type][test_case]->block->layers[16]->transform->convolution_2d, runtime, datatype);
        ck_compare_convolution_2d(torch_models_convolutional_neural_network[runtime][datatype][test_case]->conv4,
                                  models[runtime][datatype][model_type][test_case]->block->layers[19]->transform->convolution_2d, runtime, datatype);
        ck_compare_batch_normalization_2d(torch_models_convolutional_neural_network[runtime][datatype][test_case]->batch_norm1,
                                          models[runtime][datatype][model_type][test_case]->block->layers[1]->transform->batch_normalization_2d, runtime, datatype);
        ck_compare_batch_normalization_2d(torch_models_convolutional_neural_network[runtime][datatype][test_case]->batch_norm2,
                                          models[runtime][datatype][model_type][test_case]->block->layers[4]->transform->batch_normalization_2d, runtime, datatype);
        ck_compare_batch_normalization_2d(torch_models_convolutional_neural_network[runtime][datatype][test_case]->batch_norm3,
                                          models[runtime][datatype][model_type][test_case]->block->layers[7]->transform->batch_normalization_2d, runtime, datatype);
        ck_compare_batch_normalization_2d(torch_models_convolutional_neural_network[runtime][datatype][test_case]->batch_norm4,
                                          models[runtime][datatype][model_type][test_case]->block->layers[14]->transform->batch_normalization_2d, runtime, datatype);
        ck_compare_batch_normalization_2d(torch_models_convolutional_neural_network[runtime][datatype][test_case]->batch_norm5,
                                          models[runtime][datatype][model_type][test_case]->block->layers[17]->transform->batch_normalization_2d, runtime, datatype);
        break;
    case TRANSFORMER:
        ck_compare_embedding(torch_models_transformer[runtime][datatype][test_case]->token_embedding,
                             models[runtime][datatype][model_type][test_case]->block->layers[0]->transform->transformer_embedding->token_embedding, runtime, datatype);
        ck_compare_embedding(torch_models_transformer[runtime][datatype][test_case]->position_embedding,
                             models[runtime][datatype][model_type][test_case]->block->layers[0]->transform->transformer_embedding->position_embedding, runtime, datatype);
        ck_compare_linear(torch_models_transformer[runtime][datatype][test_case]->linear1,
                          models[runtime][datatype][model_type][test_case]->block->layers[2]->transform->block->layers[1]->transform->block->layers[1]->transform->linear, runtime, datatype);
        ck_compare_linear(torch_models_transformer[runtime][datatype][test_case]->linear2,
                          models[runtime][datatype][model_type][test_case]->block->layers[2]->transform->block->layers[1]->transform->block->layers[3]->transform->linear, runtime, datatype);
        ck_compare_linear(torch_models_transformer[runtime][datatype][test_case]->linear3,
                          models[runtime][datatype][model_type][test_case]->block->layers[4]->transform->linear, runtime, datatype);
        ck_compare_multihead_attention(torch_models_transformer[runtime][datatype][test_case]->multihead_attention,
                                       models[runtime][datatype][model_type][test_case]->block->layers[2]->transform->block->layers[0]->transform->block->layers[1]->transform->causal_multihead_self_attention, 
                                       runtime, datatype);
        ck_compare_layer_normalization(torch_models_transformer[runtime][datatype][test_case]->layer_norm1,
                                       models[runtime][datatype][model_type][test_case]->block->layers[2]->transform->block->layers[0]->transform->block->layers[0]->transform->layer_normalization, runtime, datatype);
        ck_compare_layer_normalization(torch_models_transformer[runtime][datatype][test_case]->layer_norm2,
                                       models[runtime][datatype][model_type][test_case]->block->layers[2]->transform->block->layers[1]->transform->block->layers[0]->transform->layer_normalization, runtime, datatype);
        ck_compare_layer_normalization(torch_models_transformer[runtime][datatype][test_case]->layer_norm3,
                                       models[runtime][datatype][model_type][test_case]->block->layers[3]->transform->layer_normalization, runtime, datatype);
        break;
    default:
        ck_abort_msg("unknown model.");
    }
}

void test_optimizer(algorithm_type_t algorithm_type)
{
    for (int i = 0; i < RUNTIMES; i++)
    {
        runtime_t runtime = (runtime_t) i;
        for (int j = 0; j < DATATYPES; j++)
        {
            datatype_t datatype = (datatype_t) j;
            for (int k = 0; k < MODELS; ++k)
            {
                model_type_t model_type = (model_type_t) k;
                for (int l = 0; l < cases(algorithm_type); ++l)
                {
                    for (int m = 0; m < iterations(algorithm_type, l); ++m)
                    {
                        torch::Tensor torch_output;
                        torch::Tensor torch_cost;
                        tensor_t *output = NULL;
                        tensor_t *cost = NULL;

                        error = zero_gradient_model(models[i][j][k][l]);
                        ck_assert_ptr_null(error);
                        error = model_forward(models[i][j][k][l], inputs[i][j][k][l], &output);
                        ck_assert_ptr_null(error);

                        switch (model_type)
                        {
                        case SINGLE_LAYER_FEED_FORWARD:
                            torch_models_single_layer_feed_forward[i][j][l]->zero_grad();
                            torch_output = torch_models_single_layer_feed_forward[i][j][l]->forward(torch_inputs[i][j][k][l]);
                            torch::nn::functional::binary_cross_entropy(torch_output, torch_outputs[i][j][k][l]).backward();
                            error = binary_cross_entropy(outputs[i][j][k][l], output, &cost);
                            ck_assert_ptr_null(error);
                            break;
                        case CONVOLUTIONAL_NEURAL_NETWORK:
                            torch_models_convolutional_neural_network[i][j][l]->zero_grad();
                            torch_output = torch_models_convolutional_neural_network[i][j][l]->forward(torch_inputs[i][j][k][l]);
                            torch::nn::functional::binary_cross_entropy(torch_output, torch_outputs[i][j][k][l]).backward();
                            error = binary_cross_entropy(outputs[i][j][k][l], output, &cost);
                            ck_assert_ptr_null(error);
                            break;
                        case TRANSFORMER:
                            torch_models_transformer[i][j][l]->zero_grad();
                            torch_output = torch_models_transformer[i][j][l]->forward(torch_inputs[i][j][k][l]);
                            // ck_assert_tensor_equiv(output, torch_to_tensor(torch_output, (runtime_t) i, (datatype_t) j));
                            torch_cost = torch::nn::functional::cross_entropy(torch_output, torch_outputs[i][j][k][l].view(-1).to(torch::kLong));
                            torch_cost.backward();
                            error = categorical_cross_entropy(outputs[i][j][k][l], output, &cost);
                            // ck_assert_tensor_equiv(cost, torch_to_tensor(torch_cost, (runtime_t) i, (datatype_t) j));
                            ck_assert_ptr_null(error);
                            break;
                        default:
                            ck_abort_msg("unknown model.");
                        }

                        error = tensor_backward(cost, NULL);
                        ck_assert_ptr_null(error);
                        error = update_model(optimizers[i][j][k][l], models[i][j][k][l]);
                        ck_assert_ptr_null(error);

                        switch (algorithm_type)
                        {
                        case STOCASTIC_GRADIENT_DESCENT:
                            torch_optimizers_sgd[i][j][k][l].step();    
                            break;
                        case RMS_PROP:
                            torch_optimizers_rms_prop[i][j][k][l].step();    
                            break;
                        case ADAM:
                            torch_optimizers_adam[i][j][k][l].step();    
                            break;
                        default:
                            ck_abort_msg("unknown optimization algorithm.");
                        }

                        ck_compare_models(runtime, datatype, model_type, l);
                    }
                }
            }
        }
    }
}

void setup_sgd(void)
{
    setup_optimizer(STOCASTIC_GRADIENT_DESCENT);
}

void teardown_sgd(void)
{
    teardown_optimizer(STOCASTIC_GRADIENT_DESCENT);
}

START_TEST(test_sgd)
{
    test_optimizer(STOCASTIC_GRADIENT_DESCENT);
}
END_TEST

void setup_rms_prop(void)
{
    setup_optimizer(RMS_PROP);
}

void teardown_rms_prop(void)
{
    teardown_optimizer(RMS_PROP);
}

START_TEST(test_rms_prop)
{
    test_optimizer(RMS_PROP);
}
END_TEST

void setup_adam(void)
{
    setup_optimizer(ADAM);
}

void teardown_adam(void)
{
    teardown_optimizer(ADAM);
}

START_TEST(test_adam)
{
    test_optimizer(ADAM);
}
END_TEST

Suite *make_optimizer_suite(void)
{
    Suite *s;
    TCase *tc_sgd;
    TCase *tc_rms_prop;
    TCase *tc_adam;

    s = suite_create("Test Optimizer Suite");

    tc_sgd = tcase_create("Test SGD Case");
    tcase_add_checked_fixture(tc_sgd, setup_sgd, teardown_sgd);
    tcase_add_test(tc_sgd, test_sgd);

    tc_rms_prop = tcase_create("Test RMS Prop Case");
    tcase_add_checked_fixture(tc_rms_prop, setup_rms_prop, teardown_rms_prop);
    tcase_add_test(tc_rms_prop, test_rms_prop);

    tc_adam = tcase_create("Test ADAM Case");
    tcase_add_checked_fixture(tc_adam, setup_adam, teardown_adam);
    tcase_add_test(tc_adam, test_adam);

    suite_add_tcase(s, tc_sgd);
    suite_add_tcase(s, tc_adam);
    suite_add_tcase(s, tc_rms_prop);

    return s;
}


int main(void) 
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_optimizer_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}