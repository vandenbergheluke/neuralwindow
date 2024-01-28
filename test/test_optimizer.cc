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
}
#include <test_helper_torch.h>

#define STOCASTIC_GRADIENT_DESCENT_CASES 2
#define RMS_PROP_CASES 2
#define ADAM_CASES 2

float32_t sgd_learning_rate_f[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    1e-3,
    1e-2,
};

float64_t sgd_learning_rate[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    1e-3,
    1e-2,
};

float32_t sgd_momentum_f[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    0.4,
    0.2,
};

float64_t sgd_momentum[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    0.4,
    0.2,
};

float32_t sgd_dampening_f[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    0.2,
    0.0,
};

float64_t sgd_dampening[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    0.2,
    0.0,
};

float32_t sgd_weight_decay_f[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    1e-4,
    1e-2,
};

float64_t sgd_weight_decay[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    1e-4,
    1e-2,
};

bool sgd_nesterov[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    false,
    true,
};

int sgd_iterations[STOCASTIC_GRADIENT_DESCENT_CASES] = {
    10,
    20,
};

float32_t rms_prop_learning_rate_f[RMS_PROP_CASES] = {
    1e-3,
    1e-3,
};

float64_t rms_prop_learning_rate[RMS_PROP_CASES] = {
    1e-3,
    1e-3,
};

float32_t rms_prop_momentum_f[RMS_PROP_CASES] = {
    0.5,
    0.0,
};

float64_t rms_prop_momentum[RMS_PROP_CASES] = {
    0.5,
    0.0,
};

float32_t rms_prop_alpha_f[RMS_PROP_CASES] = {
    0.9,
    0.9,
};

float64_t rms_prop_alpha[RMS_PROP_CASES] = {
    0.9,
    0.9,
};

float32_t rms_prop_epsilon_f[RMS_PROP_CASES] = {
    1e-8,
    1e-8,
};

float64_t rms_prop_epsilon[RMS_PROP_CASES] = {
    1e-8,
    1e-8,
};

float32_t rms_prop_weight_decay_f[RMS_PROP_CASES] = {
    1e-4,
    1e-1,
};

float64_t rms_prop_weight_decay[RMS_PROP_CASES] = {
    1e-4,
    1e-1,
};

bool rms_prop_centered[RMS_PROP_CASES] = {
    false,
    true,
};

int rms_prop_iterations[RMS_PROP_CASES] = {
    5,
    5,
};

float32_t adam_learning_rate_f[ADAM_CASES] = {
    1e-3,
    1e-3,
};

float64_t adam_learning_rate[ADAM_CASES] = {
    1e-3,
    1e-3,
};

float32_t adam_beta_1_f[ADAM_CASES] = {
    0.99,
    0.99,
};

float64_t adam_beta_1[ADAM_CASES] = {
    0.99,
    0.99,
};

float32_t adam_beta_2_f[ADAM_CASES] = {
    0.99,
    0.99,
};

float64_t adam_beta_2[ADAM_CASES] = {
    0.99,
    0.99,
};

float32_t adam_weight_decay_f[ADAM_CASES] = {
    1e-4,
    1e-3,
};

float64_t adam_weight_decay[ADAM_CASES] = {
    1e-4,
    1e-3,
};

float32_t adam_epsilon_f[ADAM_CASES] = {
    1e-5,
    1e-2,
};

float64_t adam_epsilon[ADAM_CASES] = {
    1e-5,
    1e-2,
};

int adam_iterations[ADAM_CASES] = {
    3,
    3,
};

#define MODELS 1

typedef enum model_type_t
{
    SINGLE_LAYER_FEED_FORWARD,
} model_type_t;

struct Model0Impl : torch::nn::Module
{
    torch::Tensor weight, bias;

    Model0Impl(torch::Tensor &weight, torch::Tensor &bias)
    {
        this->weight = register_parameter("weight", weight);
        this->bias = register_parameter("bias", bias);
    }

    torch::Tensor forward(torch::Tensor input) 
    {
        return torch::nn::functional::relu(torch::nn::functional::linear(input, weight.t(), bias));
    }
};
TORCH_MODULE(Model0);

nw_error_t *error = NULL;
std::vector<optimizer_t *> optimizers[RUNTIMES][DATATYPES][MODELS];
std::vector<torch::optim::SGD> torch_optimizers_sgd[RUNTIMES][DATATYPES][MODELS];
std::vector<torch::optim::RMSprop> torch_optimizers_rms_prop[RUNTIMES][DATATYPES][MODELS];
std::vector<torch::optim::Adam> torch_optimizers_adam[RUNTIMES][DATATYPES][MODELS];
std::vector<model_t *> models[RUNTIMES][DATATYPES][MODELS];
std::vector<Model0> torch_models_0[RUNTIMES][DATATYPES];
std::vector<tensor_t *> inputs[RUNTIMES][DATATYPES][MODELS];
std::vector<torch::Tensor> torch_inputs[RUNTIMES][DATATYPES][MODELS];

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

void setup_model(runtime_t runtime, datatype_t datatype, model_type_t model_type)
{
    if (model_type == SINGLE_LAYER_FEED_FORWARD)
    {
        // Torch Model
        // torch_inputs[runtime][datatype][model_case]
        torch::Tensor torch_weights;
        torch::Tensor torch_bias;
        torch::Tensor torch_input;
        switch (datatype)
        {
        case FLOAT32:
            torch_weights = torch::randn({5, 6}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
            torch_bias = torch::randn({6}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
            torch_input = torch::randn({8, 5}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
            break;
        case FLOAT64:
            torch_weights = torch::randn({5, 6}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
            torch_bias = torch::randn({6}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
            torch_input = torch::randn({8, 5}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(false));
            break;
        default:
            ck_abort_msg("unknown data type.");
        }
        torch_models_0[runtime][datatype].push_back(Model0(torch_weights, torch_bias));
        torch_inputs[runtime][datatype][model_type].push_back(torch_input);

        // NW Model
        model_t *model = NULL;
        tensor_t *weights = torch_to_tensor(torch_weights, runtime, datatype);
        tensor_t *bias = torch_to_tensor(torch_bias, runtime, datatype);
        tensor_t *input = torch_to_tensor(torch_input, runtime, datatype);

        activation_t *activation_1 = NULL;
        linear_t *linear_1 = NULL;
        transform_t *transform_1 = NULL;
        layer_t *layer_1 = NULL;
        block_t *block = NULL;
        
        error = rectified_linear_activation_create(&activation_1);
        ck_assert_ptr_null(error);
        ck_assert_ptr_nonnull(activation_1);

        error = linear_create(&linear_1, weights, bias, activation_1);
        ck_assert_ptr_null(error);
        ck_assert_ptr_nonnull(activation_1);

        error = transform_create(&transform_1, LINEAR, linear_1);
        ck_assert_ptr_null(error);
        ck_assert_ptr_nonnull(linear_1);

        error = layer_create(&layer_1, transform_1, LINEAR);
        ck_assert_ptr_null(error);
        ck_assert_ptr_nonnull(layer_1);

        error = block_create(&block, 1, layer_1);
        ck_assert_ptr_null(error);
        ck_assert_ptr_nonnull(block);

        error = model_create(&model, block);
        ck_assert_ptr_null(error);

        models[runtime][datatype][model_type].push_back(model);
        inputs[runtime][datatype][model_type].push_back(input);
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
                optimizers[i][j][k] = std::vector<optimizer_t *>(CASES);
                inputs[i][j][k].clear();
                models[i][j][k].clear();
                torch_inputs[i][j][k].clear();
                switch (model_type)
                {
                case SINGLE_LAYER_FEED_FORWARD:
                    torch_models_0[i][j].clear();
                    break;
                default:
                    ck_abort_msg("unknwown model type.");
                }
                for (int l = 0; l < CASES; ++l)
                {
                    setup_model(runtime, datatype, model_type);
                    torch::autograd::variable_list parameters;

                    switch (model_type)
                    {
                    case SINGLE_LAYER_FEED_FORWARD:
                        parameters = torch_models_0[i][j][l]->parameters();
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

                    models[i][j][k][l] = NULL;
                    optimizers[i][j][k][l] = NULL;
                    inputs[i][j][k][l] = NULL;
                }
            }
        }
    }
}

void ck_compare_models(runtime_t runtime, datatype_t datatype, model_type_t model_type, int test_case)
{
    tensor_t *torch_parameters = NULL; 

    switch (model_type)
    {
    case SINGLE_LAYER_FEED_FORWARD:
        torch_parameters = torch_to_tensor(torch_models_0[runtime][datatype][test_case]->weight, runtime, datatype);
        ck_assert_tensor_eq(models[runtime][datatype][model_type][test_case]->block->layers[0]->transform->linear->weights, torch_parameters);
        tensor_destroy(torch_parameters);
        torch_parameters = torch_to_tensor(torch_models_0[runtime][datatype][test_case]->bias, runtime, datatype);
        ck_assert_tensor_eq(models[runtime][datatype][model_type][test_case]->block->layers[0]->transform->linear->bias, torch_parameters);
        tensor_destroy(torch_parameters);
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
                        tensor_t *output = NULL;
                        tensor_t *cost = NULL;

                        error = model_forward(models[i][j][k][l], inputs[i][j][k][l], &output);
                        ck_assert_ptr_null(error);
                        error = tensor_summation(output, &cost, NULL, 0, false);
                        ck_assert_ptr_null(error);
                        error = tensor_backward(cost, NULL);
                        ck_assert_ptr_null(error);
                        error = optimizer_step(optimizers[i][j][k][l], models[i][j][k][l]);
                        ck_assert_ptr_null(error);

                        switch (model_type)
                        {
                        case SINGLE_LAYER_FEED_FORWARD:
                            torch_models_0[i][j][l]->zero_grad();
                            torch_models_0[i][j][l]->forward(torch_inputs[i][j][k][l]).sum().backward();
                            break;
                        default:
                            ck_abort_msg("unknown model.");
                        }

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
    suite_add_tcase(s, tc_rms_prop);
    suite_add_tcase(s, tc_adam);

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