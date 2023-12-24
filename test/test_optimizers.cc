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
}
#include <test_helper.h>

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

void initialize_out()
{
    out1 = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT32);
    out2 = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT32);
    out3 = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT32);
    out4 = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT32);
}

void initialize_xwm()
{
    x_torch = torch::randn({1, 4}, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));
    w_torch = torch::randn({4, 4}, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));
    b_torch = torch::zeros({1, 4}, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));
    m_torch = torch::randn({1, 4}, torch::TensorOptions().requires_grad(false).dtype(torch::kFloat32));
    w_torch_zeros = torch::zeros({4, 4}, torch::TensorOptions().requires_grad(false).dtype(torch::kFloat32));
    
    x = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT32);
    w = torch_to_tensor(w_torch, MKL_RUNTIME, FLOAT32);
    b = torch_to_tensor(b_torch, MKL_RUNTIME, FLOAT32);
    m = torch_to_tensor(m_torch, MKL_RUNTIME, FLOAT32);
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

void take_step_SGD(float32_t learning_rate, float32_t momentum, float32_t dampening, float32_t weight_decay, bool_t nesterov, int steps)
{
    initialize_xwm();

    stochastic_gradient_descent_t *SGD = NULL;

    setup_params();
    ck_assert_ptr_nonnull(model);

    error = stochastic_gradient_descent_create(&SGD,
                                            model->block,
                                            FLOAT32,
                                            &learning_rate,
                                            &momentum,
                                            &dampening,
                                            &weight_decay,
                                            nesterov);
    ck_assert_ptr_null(error);

    torch::optim::SGDOptions sgdOptions(learning_rate);
    sgdOptions.momentum(momentum)
              .dampening(dampening)
              .weight_decay(weight_decay)
              .nesterov(nesterov);

    torch::optim::SGD optim = torch::optim::SGD({x_torch, w_torch}, sgdOptions);

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

    tensor_t *x_torch_tensor = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT32);
    ck_assert_tensor_equiv(x_torch_tensor, x);
   
    tensor_t *w_torch_tensor = torch_to_tensor(w_torch, MKL_RUNTIME, FLOAT32);
    ck_assert_tensor_equiv(w_torch_tensor, w);

    tensor_destroy(x_torch_tensor);
    tensor_destroy(w_torch_tensor);
    destory();
    stochastic_gradient_descent_destroy(SGD);
}

void take_step_RMS_PROP(float32_t learning_rate, float32_t momentum, float32_t alpha, float32_t weight_decay, float32_t epsilon, bool_t centered, int steps)
{
    initialize_xwm();

    rms_prop_t *RMS = NULL;

    setup_params();
    ck_assert_ptr_nonnull(model);

    error = rms_prop_create(&RMS,
                            model->block,
                            FLOAT32,
                            &learning_rate,
                            &momentum,
                            &alpha,
                            &weight_decay,
                            &epsilon,
                            centered);
    ck_assert_ptr_null(error);

    torch::optim::RMSpropOptions rms_prop_options(learning_rate);
    rms_prop_options.momentum(momentum)
                    .alpha(alpha)
                    .weight_decay(weight_decay)
                    .eps(epsilon)
                    .centered(centered);

    torch::optim::RMSprop optim = torch::optim::RMSprop({x_torch, w_torch}, rms_prop_options);

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
        if (error)
        {
            error_print(error);
            error_destroy(error); 
        }
        ck_assert_ptr_null(error);

        error = rms_prop(RMS, w, 1);
        ck_assert_ptr_null(error);

        outNW = NULL;
        zero_grad();
    }

    tensor_t *x_torch_tensor = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT32);
    ck_assert_tensor_equiv(x_torch_tensor, x);
   
    tensor_t *w_torch_tensor = torch_to_tensor(w_torch, MKL_RUNTIME, FLOAT32);
    ck_assert_tensor_equiv(w_torch_tensor, w);

    tensor_destroy(x_torch_tensor);
    tensor_destroy(w_torch_tensor);
    destory();
    rms_prop_destroy(RMS);
}

void setup(void){}

void teardown(void){}

START_TEST(test_SGD)
{
   take_step_SGD(0.001, 0.0, 0.0, 0.0, false, 1);
}
END_TEST

START_TEST(test_sgd_high_lr)
{
   take_step_SGD(10, 0.0, 0.0, 0.0, false, 1);
}
END_TEST

START_TEST(test_sgd_wd)
{
   take_step_SGD(0.001, 0.0, 0.0, 0.1, false, 1);
}
END_TEST

START_TEST(test_sgd_high_lr_wd)
{
   take_step_SGD(10, 0.0, 0.0, 0.1, false, 1);
}
END_TEST

START_TEST(test_multistep_sgd)
{
   take_step_SGD(0.001, 0.0, 0.0, 0.0, false, 10);
}
END_TEST

START_TEST(test_multistep_sgd_high_lr)
{
   take_step_SGD(10, 0.0, 0.0, 0.0, false, 10);
}
END_TEST

START_TEST(test_multistep_sgd_wd)
{
    take_step_SGD(0.001, 0.0, 0.0, 0.1, false, 10);
}
END_TEST

START_TEST(test_multistep_sgd_high_lr_wd)
{
    take_step_SGD(9, 0.0, 0.0, 0.1, false, 10);
}
END_TEST

START_TEST(test_multistep_sgd_momentum)
{
    take_step_SGD(0.001, 0.9, 0.0, 0.0, false, 2);
}
END_TEST

START_TEST(test_multistep_sgd_high_lr_momentum)
{
    take_step_SGD(10, 0.9, 0.0, 0.0, false, 10);
}
END_TEST

START_TEST(test_multistep_sgd_momentum_wd)
{
    take_step_SGD(0.001, 0.9, 0.0, 0.1, false, 10);
}
END_TEST

START_TEST(test_multistep_sgd_high_lr_momentum_wd)
{
    take_step_SGD(10, 0.9, 0.0, 0.1, false, 10);
}
END_TEST

START_TEST(test_multistep_sgd_momentum_damp)
{
    take_step_SGD(0.001, 0.9, 0.2, 0.0, false, 2);
}
END_TEST

START_TEST(test_multistep_sgd_high_lr_momentum_damp)
{
    take_step_SGD(10, 0.9, 0.2, 0.0, false, 10);
}
END_TEST

START_TEST(test_multistep_sgd_momentum_wd_damp)
{
    take_step_SGD(0.001, 0.9, 0.2, 0.1, false, 10);
}
END_TEST

START_TEST(test_multistep_sgd_high_lr_momentum_wd_damp)
{
    take_step_SGD(9, 0.9, 0.8, 0.1, false, 10);
}
END_TEST

START_TEST(test_multistep_sgd_nesterov_momentum)
{
    take_step_SGD(0.001, 0.9, 0.0, 0.0, true, 10);
}
END_TEST

START_TEST(test_multistep_sgd_high_lr_nesterov_momentum)
{
    take_step_SGD(10, 0.9, 0.0, 0.0, true, 10);
}
END_TEST

START_TEST(test_multistep_sgd_nesterov_momentum_wd)
{
    take_step_SGD(0.001, 0.9, 0.0, 0.1, true, 10);
}
END_TEST

START_TEST(test_multistep_sgd_high_lr_nesterov_momentum_wd)
{
    take_step_SGD(9, 0.9, 0.0, 0.1, true, 10);
}
END_TEST

START_TEST(test_RMS)
{
   take_step_RMS_PROP(0.001, 0.0, 0.8, 0.0, 0.0001, false, 1);
}
END_TEST

START_TEST(test_rms_prop_high_lr)
{
    take_step_RMS_PROP(10, 0.0, 0.8, 0.0, 0.0001, false, 1);
}
END_TEST

START_TEST(test_rms_prop_wd)
{
   take_step_RMS_PROP(0.001, 0.0, 0.8, 0.1, 0.0001, false, 1);
}
END_TEST

START_TEST(test_rms_prop_high_lr_wd)
{
    take_step_RMS_PROP(10, 0.0, 0.8, 0.1, 0.0001, false, 1);
}
END_TEST

START_TEST(test_multistep_rms_prop)
{
   take_step_RMS_PROP(0.001, 0.0, 0.8, 0.0, 0.0001, false, 10);
}
END_TEST

START_TEST(test_multistep_rms_high_lr)
{
   take_step_RMS_PROP(10, 0.0, 0.8, 0.0, 0.0001, false, 10);
}
END_TEST

START_TEST(test_multistep_rms_wd)
{
   take_step_RMS_PROP(0.001, 0.0, 0.8, 0.1, 0.0001, false, 10);
}
END_TEST

START_TEST(test_multistep_rms_high_lr_wd)
{
   take_step_RMS_PROP(10, 0.0, 0.8, 0.1, 0.0001, false, 10);
}
END_TEST

START_TEST(test_multistep_rms_momentum)
{
   take_step_RMS_PROP(0.001, 0.9, 0.8, 0.0, 0.0001, false, 10);
}
END_TEST

START_TEST(test_multistep_rms_high_lr_momentum)
{
   take_step_RMS_PROP(10, 0.9, 0.8, 0.0, 0.0001, false, 10);
}
END_TEST

START_TEST(test_multistep_rms_momentum_wd)
{
   take_step_RMS_PROP(0.001, 0.9, 0.8, 0.1, 0.0001, false, 10);
}
END_TEST

START_TEST(test_multistep_rms_high_lr_momentum_wd)
{
   take_step_RMS_PROP(10, 0.9, 0.8, 0.1, 0.0001, false, 10);
}
END_TEST

Suite *make_ptimizers_suite(void)
{
    Suite *s;
    TCase *tc_unary;

    s = suite_create("Test Optimizers Suite");

    tc_unary = tcase_create("Optimizers Case");

    // tcase_add_test(tc_unary, test_SGD);
    // tcase_add_test(tc_unary, test_sgd_high_lr);
    // tcase_add_test(tc_unary, test_sgd_wd); 
    // tcase_add_test(tc_unary, test_sgd_high_lr_wd); 

    // tcase_add_test(tc_unary, test_multistep_sgd);
    // tcase_add_test(tc_unary, test_multistep_sgd_high_lr);
    // tcase_add_test(tc_unary, test_multistep_sgd_wd); 
    // tcase_add_test(tc_unary, test_multistep_sgd_high_lr_wd);

    tcase_add_test(tc_unary, test_multistep_sgd_momentum);
    tcase_add_test(tc_unary, test_multistep_sgd_high_lr_momentum);
    tcase_add_test(tc_unary, test_multistep_sgd_momentum_wd);
    tcase_add_test(tc_unary, test_multistep_sgd_high_lr_momentum_wd);

    // tcase_add_test(tc_unary, test_multistep_sgd_momentum_damp);
    // tcase_add_test(tc_unary, test_multistep_sgd_high_lr_momentum_damp);
    // tcase_add_test(tc_unary, test_multistep_sgd_momentum_wd_damp);
    // tcase_add_test(tc_unary, test_multistep_sgd_high_lr_momentum_wd_damp);

    // tcase_add_test(tc_unary, test_multistep_sgd_nesterov_momentum);
    // tcase_add_test(tc_unary, test_multistep_sgd_high_lr_nesterov_momentum);
    // tcase_add_test(tc_unary, test_multistep_sgd_nesterov_momentum_wd);
    // tcase_add_test(tc_unary, test_multistep_sgd_high_lr_nesterov_momentum_wd);

    // // //RMS PROP
    // tcase_add_test(tc_unary, test_RMS);
    // tcase_add_test(tc_unary, test_rms_prop_high_lr);
    // tcase_add_test(tc_unary, test_rms_prop_wd); 
    // tcase_add_test(tc_unary, test_rms_prop_high_lr_wd); 

    // tcase_add_test(tc_unary, test_multistep_rms_prop);
    // tcase_add_test(tc_unary, test_multistep_rms_high_lr);
    // tcase_add_test(tc_unary, test_multistep_rms_wd); 
    // tcase_add_test(tc_unary, test_multistep_rms_high_lr_wd);

    // tcase_add_test(tc_unary, test_multistep_rms_momentum);
    // tcase_add_test(tc_unary, test_multistep_rms_high_lr_momentum);
    // tcase_add_test(tc_unary, test_multistep_rms_momentum_wd);
    // tcase_add_test(tc_unary, test_multistep_rms_high_lr_momentum_wd);

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