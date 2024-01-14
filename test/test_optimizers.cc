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

nw_error_t *error = NULL;

model_t *model = NULL;
tensor_t *x = NULL; 
tensor_t *w = NULL;
tensor_t *b = NULL;
tensor_t *m = NULL;
tensor_t *out1 = NULL; 
tensor_t *out2 = NULL;
tensor_t *out3 = NULL;
tensor_t *out4 = NULL; 
tensor_t *outNW = NULL;
torch::Tensor x_torch;
torch::Tensor w_torch;
torch::Tensor b_torch;
torch::Tensor m_torch;
torch::Tensor w_torch_zeros;

void initialize_xwm(datatype_t type)
{
    switch(type) 
    {
        case FLOAT32:
            x_torch = torch::randn({1, 4}, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));
            w_torch = torch::randn({4, 4}, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));
            b_torch = torch::zeros({1, 4}, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));
            m_torch = torch::randn({1, 4}, torch::TensorOptions().requires_grad(false).dtype(torch::kFloat32));
            w_torch_zeros = torch::zeros({4, 4}, torch::TensorOptions().requires_grad(false).dtype(torch::kFloat32));
            
            x = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT32);
            w = torch_to_tensor(w_torch, MKL_RUNTIME, FLOAT32);
            b = torch_to_tensor(b_torch, MKL_RUNTIME, FLOAT32);
            m = torch_to_tensor(m_torch, MKL_RUNTIME, FLOAT32);
            break;
        case FLOAT64:
            x_torch = torch::randn({1, 4}, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat64));
            w_torch = torch::randn({4, 4}, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat64));
            b_torch = torch::zeros({1, 4}, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat64));
            m_torch = torch::randn({1, 4}, torch::TensorOptions().requires_grad(false).dtype(torch::kFloat64));
            w_torch_zeros = torch::zeros({4, 4}, torch::TensorOptions().requires_grad(false).dtype(torch::kFloat64));
            
            x = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT64);
            w = torch_to_tensor(w_torch, MKL_RUNTIME, FLOAT64);
            b = torch_to_tensor(b_torch, MKL_RUNTIME, FLOAT64);
            m = torch_to_tensor(m_torch, MKL_RUNTIME, FLOAT64);
            break;
        default:
            ck_abort_msg("unknown datatype.");
    }
}

void forwardNW()
{
    error = model_forward(model, x, &out1);
    ck_assert_ptr_null(error);

    error = tensor_logsoftmax(out1, &out2, 1);
    ck_assert_ptr_null(error);

    error = tensor_multiplication(out2, m, &out3);
    ck_assert_ptr_null(error);

    error = tensor_addition(out3, m, &out4);
    ck_assert_ptr_null(error);

    error = tensor_summation(out4, &outNW, NULL, 0, false);
    ck_assert_ptr_null(error);
}

torch::Tensor forwardTorch()
{
    torch::Tensor out = x_torch.matmul(w_torch).relu();
    out = out.log_softmax(1);
    out = out.mul(m_torch).add(m_torch).sum();
    return out;
}

void zero_grad()
{
    out1 = NULL; 
    out2 = NULL;
    out3 = NULL;
    out4 = NULL; 
}

void destory()
{   
    tensor_destroy(x);
    tensor_destroy(m);

    model_destroy(model);

    tensor_destroy(out1);
    tensor_destroy(out4);
    tensor_destroy(out3);
    tensor_destroy(out2);
    
    x = NULL; 
    w = NULL;
    b = NULL;
    m = NULL;

    out1 = NULL; 
    out2 = NULL;
    out3 = NULL;
    out4 = NULL;   

    model = NULL;
}

void setup_params()
{
    activation_t *activation_1 = NULL;
    linear_t *linear_1 = NULL;
    transform_t *transform_1 = NULL;
    layer_t *layer_1 = NULL;
    layer_t *layer_2 = NULL;
    block_t *block =NULL;
    
    error = rectified_linear_activation_create(&activation_1);
    ck_assert_ptr_null(error);
    ck_assert_ptr_nonnull(activation_1);

    error = linear_create(&linear_1, w, b, activation_1);
    ck_assert_ptr_null(error);
    ck_assert_ptr_nonnull(activation_1);

    error = transform_create(&transform_1, LINEAR, linear_1);
    ck_assert_ptr_null(error);
    ck_assert_ptr_nonnull(linear_1);

    error = layer_create(&layer_1, transform_1, LINEAR);
    ck_assert_ptr_null(error);
    ck_assert_ptr_nonnull(layer_1);

    error = block_create(&block, 1, layer_1, layer_2);
    ck_assert_ptr_null(error);
    ck_assert_ptr_nonnull(block);

    error = model_create(&model, block);
    ck_assert_ptr_null(error);
    ck_assert_ptr_nonnull(model);
}

void take_step_SGD(void *learning_rate, void *momentum, void *dampening, void *weight_decay, bool_t nesterov, int steps, datatype_t type)
{
    initialize_xwm(type);

    stochastic_gradient_descent_t *SGD = NULL;

    setup_params();
    ck_assert_ptr_nonnull(model);

    std::unique_ptr<torch::optim::SGD> optim_ptr;

    switch (type) {
        case FLOAT32: {
            error = stochastic_gradient_descent_create(&SGD,
                                                    model->block,
                                                    FLOAT32,
                                                    learning_rate,
                                                    momentum,
                                                    dampening,
                                                    weight_decay,
                                                    nesterov);

            torch::optim::SGDOptions sgdOptions(*(float32_t *)learning_rate);
            sgdOptions.momentum(*(float32_t *)momentum)
                    .dampening(*(float32_t *)dampening)
                    .weight_decay(*(float32_t *)weight_decay)
                    .nesterov(nesterov);

            optim_ptr = std::make_unique<torch::optim::SGD>(
                std::initializer_list<torch::Tensor>{x_torch, w_torch},
                sgdOptions);
            break;
        }
        case FLOAT64: {
            error = stochastic_gradient_descent_create(&SGD,
                                                    model->block,
                                                    FLOAT64,
                                                    learning_rate,
                                                    momentum,
                                                    dampening,
                                                    weight_decay,
                                                    nesterov);

            torch::optim::SGDOptions sgdOptions(*(float64_t *)learning_rate);
            sgdOptions.momentum(*(float64_t *)momentum)
                    .dampening(*(float64_t *)dampening)
                    .weight_decay(*(float64_t *)weight_decay)
                    .nesterov(nesterov);

            optim_ptr = std::make_unique<torch::optim::SGD>(
                std::initializer_list<torch::Tensor>{x_torch, w_torch},
                sgdOptions);
            break;
        }
        default:
            ck_abort_msg("unknown datatype.");
    }
    torch::optim::SGD& optim = *optim_ptr;

    for (int i = 0; i < steps; i++)
    {
        torch::Tensor out = forwardTorch();
        optim.zero_grad(); 
        out.backward();
        optim.step();
    }

    for (int j = 0; j < steps; j++) 
    {
        forwardNW();
        ck_assert_ptr_nonnull(outNW);
        
        error = tensor_backward(outNW, NULL);
        ck_assert_ptr_null(error);

        ck_assert_ptr_nonnull(x->gradient);
        error = stochastic_gradient_descent(SGD, x, 0);
        ck_assert_ptr_null(error);

        error = stochastic_gradient_descent(SGD, w, 1);
        ck_assert_ptr_null(error);

        outNW = NULL;
        zero_grad();
    }

    tensor_t *x_torch_tensor;
    tensor_t *w_torch_tensor;
    switch (type)
    {
    case FLOAT32:
        x_torch_tensor = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT32);
        w_torch_tensor = torch_to_tensor(w_torch, MKL_RUNTIME, FLOAT32);
        break;
    case FLOAT64:
        x_torch_tensor = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT64);
        w_torch_tensor = torch_to_tensor(w_torch, MKL_RUNTIME, FLOAT64);
        break;
    default:
        ck_abort_msg("unknown datatype.");
        break;
    }
    ck_assert_tensor_equiv(x, x_torch_tensor);
    ck_assert_tensor_equiv(w, w_torch_tensor);

    tensor_destroy(x_torch_tensor);
    tensor_destroy(w_torch_tensor);
    destory();
    stochastic_gradient_descent_destroy(SGD);
}

void take_step_RMS_PROP(void *learning_rate, void *momentum, void *alpha, void *weight_decay, void *epsilon, bool_t centered, int steps, datatype_t type)
{
    initialize_xwm(type);

    rms_prop_t *RMS = NULL;

    setup_params();
    ck_assert_ptr_nonnull(model);

    std::unique_ptr<torch::optim::RMSprop> optim_ptr;

    switch (type) {
        case FLOAT32: {
        error = rms_prop_create(&RMS,
                                model->block,
                                FLOAT32,
                                learning_rate,
                                momentum,
                                alpha,
                                weight_decay,
                                epsilon,
                                centered);

            torch::optim::RMSpropOptions rms_prop_options(*(float32_t *)learning_rate);
            rms_prop_options.momentum(*(float32_t *)momentum)
                            .alpha(*(float32_t *)alpha)
                            .weight_decay(*(float32_t *)weight_decay)
                            .eps(*(float32_t *)epsilon)
                            .centered(centered);

            optim_ptr = std::make_unique<torch::optim::RMSprop>(
                std::initializer_list<torch::Tensor>{x_torch, w_torch},
                rms_prop_options);
            break;
        }
        case FLOAT64: {
            error = rms_prop_create(&RMS,
                                    model->block,
                                    FLOAT64,
                                    learning_rate,
                                    momentum,
                                    alpha,
                                    weight_decay,
                                    epsilon,
                                    centered);

            torch::optim::RMSpropOptions rms_prop_options(*(float64_t *)learning_rate);
            rms_prop_options.momentum(*(float64_t *)momentum)
                            .alpha(*(float64_t *)alpha)
                            .weight_decay(*(float64_t *)weight_decay)
                            .eps(*(float64_t *)epsilon)
                            .centered(centered);

            optim_ptr = std::make_unique<torch::optim::RMSprop>(
                std::initializer_list<torch::Tensor>{x_torch, w_torch},
                rms_prop_options);
            break;
        }
        default:
            ck_abort_msg("unknown datatype.");
    }
    torch::optim::RMSprop& optim = *optim_ptr;

    for (int i = 0; i < steps; i++)
    {
        torch::Tensor out = forwardTorch();
        optim.zero_grad(); 
        out.backward();
        optim.step();
    }
    
    for (int j = 0; j < steps; j++) 
    {
        forwardNW();
        ck_assert_ptr_nonnull(outNW);
        
        error = tensor_backward(outNW, NULL);
        ck_assert_ptr_null(error);

        error = rms_prop(RMS, x, 0);
        ck_assert_ptr_null(error);

        error = rms_prop(RMS, w, 1);
        ck_assert_ptr_null(error);

        outNW = NULL;
        zero_grad();
    }

    tensor_t *x_torch_tensor;
    tensor_t *w_torch_tensor;
    switch (type)
    {
    case FLOAT32:
        x_torch_tensor = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT32);
        w_torch_tensor = torch_to_tensor(w_torch, MKL_RUNTIME, FLOAT32);
        break;
    case FLOAT64:
        x_torch_tensor = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT64);
        w_torch_tensor = torch_to_tensor(w_torch, MKL_RUNTIME, FLOAT64);
        break;
    default:
        ck_abort_msg("unknown datatype.");
        break;
    }

    tensor_destroy(x_torch_tensor);
    tensor_destroy(w_torch_tensor);
    destory();
    rms_prop_destroy(RMS);
}

void take_step_ADAM(void *learning_rate, void *beta_1, void * beta_2, void * weight_decay, void * epsilon, int steps, datatype_t type)
{
    initialize_xwm(type);

    adam_t *adam_optimizer = NULL;

    setup_params();
    ck_assert_ptr_nonnull(model);



    std::unique_ptr<torch::optim::Adam> optim_ptr;

    switch (type) {
        case FLOAT32: {
            error = adam_create(&adam_optimizer,
                    model->block,
                    FLOAT32,
                    learning_rate,
                    beta_1,
                    beta_2,
                    weight_decay,
                    epsilon);

            torch::optim::AdamOptions adam_options(*(float32_t *)learning_rate);
            adam_options.betas({*(float32_t *)beta_1, *(float32_t *)beta_2})
                        .weight_decay(*(float32_t *)weight_decay)
                        .eps(*(float32_t *)epsilon);

            optim_ptr = std::make_unique<torch::optim::Adam>(
                std::initializer_list<torch::Tensor>{x_torch, w_torch},
                adam_options);
            break;
        }
        case FLOAT64: {
            error = adam_create(&adam_optimizer,
                    model->block,
                    FLOAT64,
                    learning_rate,
                    beta_1,
                    beta_2,
                    weight_decay,
                    epsilon);

            torch::optim::AdamOptions adam_options(*(float64_t *)learning_rate);
            adam_options.betas({*(float64_t *)beta_1, *(float64_t *)beta_2})
                        .weight_decay(*(float64_t *)weight_decay)
                        .eps(*(float64_t *)epsilon);

            optim_ptr = std::make_unique<torch::optim::Adam>(
                std::initializer_list<torch::Tensor>{x_torch, w_torch},
                adam_options);
            break;
        }
        default:
            ck_abort_msg("unknown datatype.");
    }
    torch::optim::Adam& optim = *optim_ptr;

    for (int i = 0; i < steps; i++)
    {
        torch::Tensor out = forwardTorch();
        optim.zero_grad(); 
        out.backward();
        optim.step();
    }

    for (int j = 0; j < steps; j++) 
    {
        forwardNW();
        ck_assert_ptr_nonnull(outNW);
        
        error = tensor_backward(outNW, NULL);
        ck_assert_ptr_null(error);

        error = adam(adam_optimizer, x, 0);
        ck_assert_ptr_null(error);

        error = adam(adam_optimizer, w, 1);
        ck_assert_ptr_null(error);

        outNW = NULL;
        zero_grad();
        adam_optimizer->iteration++;
    }

    tensor_t *x_torch_tensor;
    tensor_t *w_torch_tensor;
    switch (type)
    {
    case FLOAT32:
        x_torch_tensor = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT32);
        w_torch_tensor = torch_to_tensor(w_torch, MKL_RUNTIME, FLOAT32);
        break;
    case FLOAT64:
        x_torch_tensor = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT64);
        w_torch_tensor = torch_to_tensor(w_torch, MKL_RUNTIME, FLOAT64);
        break;
    default:
        ck_abort_msg("unknown datatype.");
        break;
    }
    ck_assert_tensor_equiv(x, x_torch_tensor);
    ck_assert_tensor_equiv(w, w_torch_tensor);

    tensor_destroy(x_torch_tensor);
    tensor_destroy(w_torch_tensor);
    destory();
    adam_destroy(adam_optimizer);
}

void setup(void){}

void teardown(void){}

START_TEST(test_SGD)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.0;
    float32_t damp_32 = 0.0;
    float32_t wd_32 = 0.0;
   take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, false, 1, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.0;
    float64_t damp_64 = 0.0;
    float64_t wd_64 = 0.0;
   take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, false, 1, FLOAT64);
}
END_TEST

START_TEST(test_sgd_high_lr)
{
    float32_t lr_32 = 10.0;
    float32_t momentum_32 = 0.0;
    float32_t damp_32 = 0.0;
    float32_t wd_32 = 0.0;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, false, 1, FLOAT32);

    float64_t lr_64 = 10.0;
    float64_t momentum_64 = 0.0;
    float64_t damp_64 = 0.0;
    float64_t wd_64 = 0.0;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, false, 1, FLOAT64);
}
END_TEST

START_TEST(test_sgd_wd)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.0;
    float32_t damp_32 = 0.0;
    float32_t wd_32 = 0.1;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, false, 1, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.0;
    float64_t damp_64 = 0.0;
    float64_t wd_64 = 0.1;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, false, 1, FLOAT64);
}
END_TEST

START_TEST(test_sgd_high_lr_wd)
{
    float32_t lr_32 = 10.0;
    float32_t momentum_32 = 0.0;
    float32_t damp_32 = 0.0;
    float32_t wd_32 = 0.1;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, false, 1, FLOAT32);

    float64_t lr_64 = 10.0;
    float64_t momentum_64 = 0.0;
    float64_t damp_64 = 0.0;
    float64_t wd_64 = 0.1;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, false, 1, FLOAT64);
}
END_TEST

START_TEST(test_multistep_sgd)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.0;
    float32_t damp_32 = 0.0;
    float32_t wd_32 = 0.0;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, false, 10, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.0;
    float64_t damp_64 = 0.0;
    float64_t wd_64 = 0.0;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, false, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_sgd_high_lr)
{
    float32_t lr_32 = 10.0;
    float32_t momentum_32 = 0.0;
    float32_t damp_32 = 0.0;
    float32_t wd_32 = 0.0;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, false, 10, FLOAT32);

    float64_t lr_64 = 10.0;
    float64_t momentum_64 = 0.0;
    float64_t damp_64 = 0.0;
    float64_t wd_64 = 0.0;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, false, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_sgd_wd)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.0;
    float32_t damp_32 = 0.0;
    float32_t wd_32 = 0.1;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, false, 10, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.0;
    float64_t damp_64 = 0.0;
    float64_t wd_64 = 0.1;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, false, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_sgd_high_lr_wd)
{
    float32_t lr_32 = 9.0;
    float32_t momentum_32 = 0.0;
    float32_t damp_32 = 0.0;
    float32_t wd_32 = 0.1;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, false, 10, FLOAT32);

    float64_t lr_64 = 9.0;
    float64_t momentum_64 = 0.0;
    float64_t damp_64 = 0.0;
    float64_t wd_64 = 0.1;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, false, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_sgd_momentum)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.9;
    float32_t damp_32 = 0.0;
    float32_t wd_32 = 0.0;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, false, 2, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.9;
    float64_t damp_64 = 0.0;
    float64_t wd_64 = 0.0;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, false, 2, FLOAT64);
}
END_TEST

START_TEST(test_multistep_sgd_high_lr_momentum)
{
    float32_t lr_32 = 10.0;
    float32_t momentum_32 = 0.9;
    float32_t damp_32 = 0.0;
    float32_t wd_32 = 0.0;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, false, 10, FLOAT32);

    float64_t lr_64 = 10.0;
    float64_t momentum_64 = 0.9;
    float64_t damp_64 = 0.0;
    float64_t wd_64 = 0.0;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, false, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_sgd_momentum_wd)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.9;
    float32_t damp_32 = 0.0;
    float32_t wd_32 = 0.1;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, false, 10, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.9;
    float64_t damp_64 = 0.0;
    float64_t wd_64 = 0.1;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, false, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_sgd_high_lr_momentum_wd)
{
    float32_t lr_32 = 10.0;
    float32_t momentum_32 = 0.9;
    float32_t damp_32 = 0.0;
    float32_t wd_32 = 0.2;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, false, 10, FLOAT32);

    float64_t lr_64 = 10.0;
    float64_t momentum_64 = 0.9;
    float64_t damp_64 = 0.0;
    float64_t wd_64 = 0.2;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, false, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_sgd_momentum_damp)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.9;
    float32_t damp_32 = 0.2;
    float32_t wd_32 = 0.0;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, false, 2, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.9;
    float64_t damp_64 = 0.2;
    float64_t wd_64 = 0.0;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, false, 2, FLOAT64);
}
END_TEST

START_TEST(test_multistep_sgd_high_lr_momentum_damp)
{
    float32_t lr_32 = 10.0;
    float32_t momentum_32 = 0.9;
    float32_t damp_32 = 0.2;
    float32_t wd_32 = 0.0;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, false, 10, FLOAT32);

    float64_t lr_64 = 10.0;
    float64_t momentum_64 = 0.9;
    float64_t damp_64 = 0.2;
    float64_t wd_64 = 0.0;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, false, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_sgd_momentum_wd_damp)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.9;
    float32_t damp_32 = 0.2;
    float32_t wd_32 = 0.1;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, false, 10, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.9;
    float64_t damp_64 = 0.2;
    float64_t wd_64 = 0.1;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, false, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_sgd_high_lr_momentum_wd_damp)
{
    float32_t lr_32 = 9.0;
    float32_t momentum_32 = 0.9;
    float32_t damp_32 = 0.8;
    float32_t wd_32 = 0.1;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, false, 10, FLOAT32);

    float64_t lr_64 = 9.0;
    float64_t momentum_64 = 0.9;
    float64_t damp_64 = 0.8;
    float64_t wd_64 = 0.1;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, false, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_sgd_nesterov_momentum)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.9;
    float32_t damp_32 = 0.0;
    float32_t wd_32 = 0.0;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, true, 10, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.9;
    float64_t damp_64 = 0.0;
    float64_t wd_64 = 0.0;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, true, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_sgd_high_lr_nesterov_momentum)
{
    float32_t lr_32 = 10.0;
    float32_t momentum_32 = 0.9;
    float32_t damp_32 = 0.0;
    float32_t wd_32 = 0.0;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, true, 10, FLOAT32);

    float64_t lr_64 = 10.0;
    float64_t momentum_64 = 0.9;
    float64_t damp_64 = 0.0;
    float64_t wd_64 = 0.0;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, true, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_sgd_nesterov_momentum_wd)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.9;
    float32_t damp_32 = 0.0;
    float32_t wd_32 = 0.1;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, true, 10, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.9;
    float64_t damp_64 = 0.0;
    float64_t wd_64 = 0.1;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, true, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_sgd_high_lr_nesterov_momentum_wd)
{
    float32_t lr_32 = 9.0;
    float32_t momentum_32 = 0.9;
    float32_t damp_32 = 0.0;
    float32_t wd_32 = 0.1;
    take_step_SGD(&lr_32, &momentum_32, &damp_32, &wd_32, true, 10, FLOAT32);

    float64_t lr_64 = 9.0;
    float64_t momentum_64 = 0.9;
    float64_t damp_64 = 0.0;
    float64_t wd_64 = 0.1;
    take_step_SGD(&lr_64, &momentum_64, &damp_64, &wd_64, true, 10, FLOAT64);
}
END_TEST

// RMS PROP
START_TEST(test_RMS)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.0;
    float32_t alpha_32 = 0.8;
    float32_t wd_32 = 0.0;
    float32_t epsilon_32 = 0.0;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, false, 1, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.0;
    float64_t alpha_64 = 0.8;
    float64_t wd_64 = 0.0;
    float64_t epsilon_64 = 0.0;
   take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, false, 1, FLOAT64);
}
END_TEST

START_TEST(test_rms_prop_high_lr)
{
    float32_t lr_32 = 10.0;
    float32_t momentum_32 = 0.0;
    float32_t alpha_32 = 0.8;
    float32_t wd_32 = 0.0;
    float32_t epsilon_32 = 0.0001;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, false, 1, FLOAT32);

    float64_t lr_64 = 10.0;
    float64_t momentum_64 = 0.0;
    float64_t alpha_64 = 0.8;
    float64_t wd_64 = 0.0;
    float64_t epsilon_64 = 0.0001;

    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, false, 1, FLOAT64);

}
END_TEST

START_TEST(test_rms_prop_wd)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.0;
    float32_t alpha_32 = 0.8;
    float32_t wd_32 = 0.1;
    float32_t epsilon_32 = 0.0001;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, false, 1, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.0;
    float64_t alpha_64 = 0.8;
    float64_t wd_64 = 0.1;
    float64_t epsilon_64 = 0.0001;

    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, false, 1, FLOAT64);
}
END_TEST

START_TEST(test_rms_prop_high_lr_wd)
{
    float32_t lr_32 = 10.0;
    float32_t momentum_32 = 0.0;
    float32_t alpha_32 = 0.8;
    float32_t wd_32 = 0.1;
    float32_t epsilon_32 = 0.0001;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, false, 1, FLOAT32);

    float64_t lr_64 = 10.0;
    float64_t momentum_64 = 0.0;
    float64_t alpha_64 = 0.8;
    float64_t wd_64 = 0.1;
    float64_t epsilon_64 = 0.0001;
    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, false, 1, FLOAT64);

}
END_TEST

START_TEST(test_multistep_rms_prop)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.0;
    float32_t alpha_32 = 0.8;
    float32_t wd_32 = 0.0;
    float32_t epsilon_32 = 0.0001;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, false, 10, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.0;
    float64_t alpha_64 = 0.8;
    float64_t wd_64 = 0.0;
    float64_t epsilon_64 = 0.0001;
    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, false, 10, FLOAT64);

}
END_TEST

START_TEST(test_multistep_rms_high_lr)
{
    float32_t lr_32 = 10.0;
    float32_t momentum_32 = 0.0;
    float32_t alpha_32 = 0.8;
    float32_t wd_32 = 0.0;
    float32_t epsilon_32 = 0.0001;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, false, 10, FLOAT32);

    float64_t lr_64 = 10.0;
    float64_t momentum_64 = 0.0;
    float64_t alpha_64 = 0.8;
    float64_t wd_64 = 0.0;
    float64_t epsilon_64 = 0.0001;
    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, false, 10, FLOAT64);

}
END_TEST

START_TEST(test_multistep_rms_wd)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.0;
    float32_t alpha_32 = 0.8;
    float32_t wd_32 = 0.1;
    float32_t epsilon_32 = 0.0001;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, false, 10, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.0;
    float64_t alpha_64 = 0.8;
    float64_t wd_64 = 0.1;
    float64_t epsilon_64 = 0.0001;
    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, false, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_rms_high_lr_wd)
{
    float32_t lr_32 = 10.0;
    float32_t momentum_32 = 0.0;
    float32_t alpha_32 = 0.8;
    float32_t wd_32 = 0.1;
    float32_t epsilon_32 = 0.0001;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, false, 10, FLOAT32);

    float64_t lr_64 = 10.0;
    float64_t momentum_64 = 0.0;
    float64_t alpha_64 = 0.8;
    float64_t wd_64 = 0.1;
    float64_t epsilon_64 = 0.0001;
    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, false, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_rms_momentum)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.9;
    float32_t alpha_32 = 0.8;
    float32_t wd_32 = 0.0;
    float32_t epsilon_32 = 0.0001;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, false, 10, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.9;
    float64_t alpha_64 = 0.8;
    float64_t wd_64 = 0.0;
    float64_t epsilon_64 = 0.0001;
    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, false, 10, FLOAT64);

}
END_TEST

START_TEST(test_multistep_rms_high_lr_momentum)
{
    float32_t lr_32 = 10.0;
    float32_t momentum_32 = 0.9;
    float32_t alpha_32 = 0.8;
    float32_t wd_32 = 0.0;
    float32_t epsilon_32 = 0.0001;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, false, 10, FLOAT32);  

    float64_t lr_64 = 10.0;
    float64_t momentum_64 = 0.9;
    float64_t alpha_64 = 0.8;
    float64_t wd_64 = 0.0;
    float64_t epsilon_64 = 0.0001;
    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, false, 10, FLOAT64);

}
END_TEST

START_TEST(test_multistep_rms_momentum_wd)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.9;
    float32_t alpha_32 = 0.8;
    float32_t wd_32 = 0.1;
    float32_t epsilon_32 = 0.0001;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, false, 10, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.9;
    float64_t alpha_64 = 0.8;
    float64_t wd_64 = 0.1;
    float64_t epsilon_64 = 0.0001;
    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, false, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_rms_high_lr_momentum_wd)
{
    float32_t lr_32 = 10.0;
    float32_t momentum_32 = 0.9;
    float32_t alpha_32 = 0.8;
    float32_t wd_32 = 0.1;
    float32_t epsilon_32 = 0.0001;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, false, 10, FLOAT32);

    float64_t lr_64 = 10.0;
    float64_t momentum_64 = 0.9;
    float64_t alpha_64 = 0.8;
    float64_t wd_64 = 0.1;
    float64_t epsilon_64 = 0.0001;
    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, false, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_rms_centered)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.0;
    float32_t alpha_32 = 0.9;
    float32_t wd_32 = 0.1;
    float32_t epsilon_32 = 0.0001;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, true, 10, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.0;
    float64_t alpha_64 = 0.9;
    float64_t wd_64 = 0.1;
    float64_t epsilon_64 = 0.0001;
    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, true, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_rms_high_lr_centered)
{
    float32_t lr_32 = 10.0;
    float32_t momentum_32 = 0.0;
    float32_t alpha_32 = 0.9;
    float32_t wd_32 = 0.1;
    float32_t epsilon_32 = 0.0001;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, true, 10, FLOAT32);

    float64_t lr_64 = 10.0;
    float64_t momentum_64 = 0.0;
    float64_t alpha_64 = 0.9;
    float64_t wd_64 = 0.1;
    float64_t epsilon_64 = 0.0001;
    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, true, 10, FLOAT64);

}
END_TEST

START_TEST(test_multistep_rms_high_lr_centered_wd)
{
    float32_t lr_32 = 9.0;
    float32_t momentum_32 = 0.0;
    float32_t alpha_32 = 0.9;
    float32_t wd_32 = 0.1;
    float32_t epsilon_32 = 0.0001;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, true, 10, FLOAT32);

    float64_t lr_64 = 9.0;
    float64_t momentum_64 = 0.0;
    float64_t alpha_64 = 0.9;
    float64_t wd_64 = 0.1;
    float64_t epsilon_64 = 0.0001;
    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, true, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_rms_centered_wd)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.0;
    float32_t alpha_32 = 0.9;
    float32_t wd_32 = 0.1;
    float32_t epsilon_32 = 0.0001;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, true, 10, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.0;
    float64_t alpha_64 = 0.9;
    float64_t wd_64 = 0.1;
    float64_t epsilon_64 = 0.0001;
    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, true, 10, FLOAT64);

}
END_TEST

START_TEST(test_multistep_rms_momentum_centered)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.9;
    float32_t alpha_32 = 0.9;
    float32_t wd_32 = 0.0;
    float32_t epsilon_32 = 0.0001;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, true, 10, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.9;
    float64_t alpha_64 = 0.9;
    float64_t wd_64 = 0.0;
    float64_t epsilon_64 = 0.0001;
    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, true, 10, FLOAT64);

}
END_TEST

START_TEST(test_multistep_rms_high_lr_momentum_centered)
{
    float32_t lr_32 = 10.0;
    float32_t momentum_32 = 0.9;
    float32_t alpha_32 = 0.9;
    float32_t wd_32 = 0.0;
    float32_t epsilon_32 = 0.0001;

    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, true, 10, FLOAT32);

    float64_t lr_64 = 10.0;
    float64_t momentum_64 = 0.9;
    float64_t alpha_64 = 0.9;
    float64_t wd_64 = 0.0;
    float64_t epsilon_64 = 0.0001;

    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, true, 10, FLOAT64);

}
END_TEST

START_TEST(test_multistep_rms_high_lr_momentum_wd_centered)
{
    float32_t lr_32 = 10.0;
    float32_t momentum_32 = 0.9;
    float32_t alpha_32 = 0.9;
    float32_t wd_32 = 0.1;
    float32_t epsilon_32 = 0.0001;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, true, 10, FLOAT32);

    float64_t lr_64 = 10.0;
    float64_t momentum_64 = 0.9;
    float64_t alpha_64 = 0.9;
    float64_t wd_64 = 0.1;
    float64_t epsilon_64 = 0.0001;
    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, true, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_rms_momentum_wd_centered)
{
    float32_t lr_32 = 0.001;
    float32_t momentum_32 = 0.9;
    float32_t alpha_32 = 0.9;
    float32_t wd_32 = 0.1;
    float32_t epsilon_32 = 0.0001;
    take_step_RMS_PROP(&lr_32, &momentum_32, &alpha_32, &wd_32, &epsilon_32, true, 10, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t momentum_64 = 0.9;
    float64_t alpha_64 = 0.9;
    float64_t wd_64 = 0.1;
    float64_t epsilon_64 = 0.0001;
    take_step_RMS_PROP(&lr_64, &momentum_64, &alpha_64, &wd_64, &epsilon_64, true, 10, FLOAT64);
}
END_TEST

START_TEST(test_adam)
{
   float32_t lr_32 = 0.001;
   float32_t beta1_32 = 0.9;
   float32_t beta2_32 = 0.995;
   float32_t wd_32 = 0.0;
   float32_t epsilon_32 = 0.0001;
   take_step_ADAM(&lr_32, &beta1_32, &beta2_32, &wd_32, &epsilon_32, 1, FLOAT32);

   float64_t lr_64 = 0.001;
   float64_t beta1_64 = 0.9;
   float64_t beta2_64 = 0.995;
   float64_t wd_64 = 0.0;
   float64_t epsilon_64 = 0.0001;
   take_step_ADAM(&lr_64, &beta1_64, &beta2_64, &wd_64, &epsilon_64, 1, FLOAT64);
}
END_TEST

START_TEST(test_adam_high_lr)
{
    float32_t lr_32 = 10.0;
    float32_t beta1_32 = 0.9;
    float32_t beta2_32 = 0.995;
    float32_t wd_32 = 0.0;
    float32_t epsilon_32 = 0.0001;
    take_step_ADAM(&lr_32, &beta1_32, &beta2_32, &wd_32, &epsilon_32, 1, FLOAT32);
    
    float64_t lr_64 = 10.0;
    float64_t beta1_64 = 0.9;
    float64_t beta2_64 = 0.995;
    float64_t wd_64 = 0.0;
    float64_t epsilon_64 = 0.0001;
    take_step_ADAM(&lr_64, &beta1_64, &beta2_64, &wd_64, &epsilon_64, 1, FLOAT64);

}
END_TEST

START_TEST(test_adam_wd)
{
    float32_t lr_32 = 0.001;
    float32_t beta1_32 = 0.9;
    float32_t beta2_32 = 0.995;
    float32_t wd_32 = 0.1;
    float32_t epsilon_32 = 0.0001;
    take_step_ADAM(&lr_32, &beta1_32, &beta2_32, &wd_32, &epsilon_32, 1, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t beta1_64 = 0.9;
    float64_t beta2_64 = 0.995;
    float64_t wd_64 = 0.1;
    float64_t epsilon_64 = 0.0001;
    take_step_ADAM(&lr_64, &beta1_64, &beta2_64, &wd_64, &epsilon_64, 1, FLOAT64);

}
END_TEST

START_TEST(test_adam_high_lr_wd)
{
    float32_t lr_32 = 10.0;
    float32_t beta1_32 = 0.9;
    float32_t beta2_32 = 0.995;
    float32_t wd_32 = 0.1;
    float32_t epsilon_32 = 0.0001;
    take_step_ADAM(&lr_32, &beta1_32, &beta2_32, &wd_32, &epsilon_32, 1, FLOAT32);

    float64_t lr_64 = 10.0;
    float64_t beta1_64 = 0.9;
    float64_t beta2_64 = 0.995;
    float64_t wd_64 = 0.1;
    float64_t epsilon_64 = 0.0001;
    take_step_ADAM(&lr_64, &beta1_64, &beta2_64, &wd_64, &epsilon_64, 1, FLOAT64);
}
END_TEST

START_TEST(test_multistep_adam)
{
    float32_t lr_32 = 0.001;
    float32_t beta1_32 = 0.9;
    float32_t beta2_32 = 0.995;
    float32_t wd_32 = 0.0;
    float32_t epsilon_32 = 0.0001;
    take_step_ADAM(&lr_32, &beta1_32, &beta2_32, &wd_32, &epsilon_32, 10, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t beta1_64 = 0.9;
    float64_t beta2_64 = 0.995;
    float64_t wd_64 = 0.0;
    float64_t epsilon_64 = 0.0001;
    take_step_ADAM(&lr_64, &beta1_64, &beta2_64, &wd_64, &epsilon_64, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_adam_high_lr)
{
    float32_t lr_32 = 10.0;
    float32_t beta1_32 = 0.9;
    float32_t beta2_32 = 0.995;
    float32_t wd_32 = 0.0;
    float32_t epsilon_32 = 0.0001;
    take_step_ADAM(&lr_32, &beta1_32, &beta2_32, &wd_32, &epsilon_32, 10, FLOAT32);

    float64_t lr_64 = 10.0;
    float64_t beta1_64 = 0.9;
    float64_t beta2_64 = 0.995;
    float64_t wd_64 = 0.0;
    float64_t epsilon_64 = 0.0001;
    take_step_ADAM(&lr_64, &beta1_64, &beta2_64, &wd_64, &epsilon_64, 10, FLOAT64);

}
END_TEST

START_TEST(test_multistep_adam_wd)
{
    float32_t lr_32 = 0.001;
    float32_t beta1_32 = 0.9;
    float32_t beta2_32 = 0.995;
    float32_t wd_32 = 0.1;
    float32_t epsilon_32 = 0.0001;
    take_step_ADAM(&lr_32, &beta1_32, &beta2_32, &wd_32, &epsilon_32, 10, FLOAT32);

    float64_t lr_64 = 0.001;
    float64_t beta1_64 = 0.9;
    float64_t beta2_64 = 0.995;
    float64_t wd_64 = 0.1;
    float64_t epsilon_64 = 0.0001;
    take_step_ADAM(&lr_64, &beta1_64, &beta2_64, &wd_64, &epsilon_64, 10, FLOAT64);
}
END_TEST

START_TEST(test_multistep_adam_high_lr_wd)
{
    float32_t lr_32 = 10.0;
    float32_t beta1_32 = 0.9;
    float32_t beta2_32 = 0.995;
    float32_t wd_32 = 0.1;
    float32_t epsilon_32 = 0.0001;
    take_step_ADAM(&lr_32, &beta1_32, &beta2_32, &wd_32, &epsilon_32, 10, FLOAT32);

    float64_t lr_64 = 10.0;
    float64_t beta1_64 = 0.9;
    float64_t beta2_64 = 0.995;
    float64_t wd_64 = 0.1;
    float64_t epsilon_64 = 0.0001;
    take_step_ADAM(&lr_64, &beta1_64, &beta2_64, &wd_64, &epsilon_64, 10, FLOAT64);
}
END_TEST

Suite *make_ptimizers_suite(void)
{
    Suite *s;
    TCase *tc_unary;

    s = suite_create("Test Optimizers Suite");

    tc_unary = tcase_create("Optimizers Case");

    // SGD
    tcase_add_test(tc_unary, test_SGD);
    tcase_add_test(tc_unary, test_sgd_high_lr);
    tcase_add_test(tc_unary, test_sgd_wd); 
    tcase_add_test(tc_unary, test_sgd_high_lr_wd); 

    tcase_add_test(tc_unary, test_multistep_sgd);
    tcase_add_test(tc_unary, test_multistep_sgd_high_lr);
    tcase_add_test(tc_unary, test_multistep_sgd_wd); 
    tcase_add_test(tc_unary, test_multistep_sgd_high_lr_wd);

    tcase_add_test(tc_unary, test_multistep_sgd_momentum);
    tcase_add_test(tc_unary, test_multistep_sgd_high_lr_momentum);
    tcase_add_test(tc_unary, test_multistep_sgd_momentum_wd);
    tcase_add_test(tc_unary, test_multistep_sgd_high_lr_momentum_wd);

    tcase_add_test(tc_unary, test_multistep_sgd_momentum_damp);
    tcase_add_test(tc_unary, test_multistep_sgd_high_lr_momentum_damp);
    tcase_add_test(tc_unary, test_multistep_sgd_momentum_wd_damp);
    tcase_add_test(tc_unary, test_multistep_sgd_high_lr_momentum_wd_damp);

    tcase_add_test(tc_unary, test_multistep_sgd_nesterov_momentum);
    tcase_add_test(tc_unary, test_multistep_sgd_high_lr_nesterov_momentum);
    tcase_add_test(tc_unary, test_multistep_sgd_nesterov_momentum_wd);
    tcase_add_test(tc_unary, test_multistep_sgd_high_lr_nesterov_momentum_wd);

    // RMS PROP
    tcase_add_test(tc_unary, test_RMS);
    tcase_add_test(tc_unary, test_rms_prop_high_lr);
    tcase_add_test(tc_unary, test_rms_prop_wd); 
    tcase_add_test(tc_unary, test_rms_prop_high_lr_wd); 

    tcase_add_test(tc_unary, test_multistep_rms_prop);
    tcase_add_test(tc_unary, test_multistep_rms_high_lr);
    tcase_add_test(tc_unary, test_multistep_rms_wd); 
    tcase_add_test(tc_unary, test_multistep_rms_high_lr_wd);

    tcase_add_test(tc_unary, test_multistep_rms_momentum);
    tcase_add_test(tc_unary, test_multistep_rms_high_lr_momentum);
    tcase_add_test(tc_unary, test_multistep_rms_high_lr_momentum_wd);
    tcase_add_test(tc_unary, test_multistep_rms_momentum_wd);

    tcase_add_test(tc_unary, test_multistep_rms_centered);
    tcase_add_test(tc_unary, test_multistep_rms_high_lr_centered);
    tcase_add_test(tc_unary, test_multistep_rms_centered_wd);
    tcase_add_test(tc_unary, test_multistep_rms_high_lr_centered_wd);

    tcase_add_test(tc_unary, test_multistep_rms_momentum_centered);
    tcase_add_test(tc_unary, test_multistep_rms_high_lr_momentum_centered);
    tcase_add_test(tc_unary, test_multistep_rms_high_lr_momentum_wd_centered);
    tcase_add_test(tc_unary, test_multistep_rms_momentum_wd_centered);

    // ADAM
    tcase_add_test(tc_unary, test_adam);
    tcase_add_test(tc_unary, test_adam_high_lr);
    tcase_add_test(tc_unary, test_adam_wd); 
    tcase_add_test(tc_unary, test_adam_high_lr_wd); 

    tcase_add_test(tc_unary, test_multistep_adam); 
    tcase_add_test(tc_unary, test_multistep_adam_high_lr);
    tcase_add_test(tc_unary, test_multistep_adam_wd); 
    tcase_add_test(tc_unary, test_multistep_adam_high_lr_wd); 

    suite_add_tcase(s, tc_unary);

    return s;
}

int main(void) 
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_ptimizers_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}