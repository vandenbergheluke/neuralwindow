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
}
#include <test_helper.h>

nw_error_t *error = NULL;

tensor_t *x = NULL; 
tensor_t *w = NULL;
tensor_t *m = NULL;
tensor_t *out1 = NULL; 
tensor_t *out2 = NULL;
tensor_t *out3 = NULL;
tensor_t *out4 = NULL; 
tensor_t *out5 = NULL;
tensor_t *outNW = NULL;
torch::Tensor x_torch;
torch::Tensor w_torch;
torch::Tensor m_torch;

void initialize_out()
{
    out1 = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT32);
    out2 = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT32);
    out3 = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT32);
    out4 = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT32);
    out5 = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT32);
}

void initialize_xwm()
{
    x_torch = torch::randn({1, 4}, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));
    w_torch = torch::randn({4, 4}, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));
    m_torch = torch::randn({1, 4}, torch::TensorOptions().requires_grad(false).dtype(torch::kFloat32));
    
    x = torch_to_tensor(x_torch, MKL_RUNTIME, FLOAT32);
    w = torch_to_tensor(w_torch, MKL_RUNTIME, FLOAT32);
    m = torch_to_tensor(m_torch, MKL_RUNTIME, FLOAT32);

    initialize_out();
}

void forwardNW()
{
    error = tensor_matrix_multiplication(x, w, &out1);
    if (error) goto cleanup;

    error = tensor_rectified_linear(out1, &out2);
    if (error) goto cleanup;

    error = tensor_logsoftmax(out2, &out3, 1);
    if (error) goto cleanup;

    error = tensor_multiplication(out3, m, &out4);
    if (error) goto cleanup;

    error = tensor_addition(out4, m, &out5);
    if (error) goto cleanup;

    error = tensor_summation(out5, &outNW, NULL, 0, false);
    if (error) goto cleanup;

    return;

cleanup:
    tensor_destroy(x);
    tensor_destroy(w);
    tensor_destroy(m);

    if (out1 != NULL) {tensor_destroy(out1); out1 = NULL;}
    if (out2 != NULL) {tensor_destroy(out2); out2 = NULL;}
    if (out3 != NULL) {tensor_destroy(out3); out3 = NULL;}
    if (out4 != NULL) {tensor_destroy(out4); out4 = NULL;}
    if (out5 != NULL) {tensor_destroy(out5); out5 = NULL;}

    error_print(error);
    error_destroy(error); 
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
    tensor_destroy(out1);
    tensor_destroy(out2);
    tensor_destroy(out3);
    tensor_destroy(out4);
    tensor_destroy(out5);

    out1 = NULL; 
    out2 = NULL;
    out3 = NULL;
    out4 = NULL; 
    out5 = NULL;

    initialize_out();
}

void destory()
{   
    tensor_destroy(x);
    tensor_destroy(w);
    tensor_destroy(m); 

    tensor_destroy(out1);
    tensor_destroy(out2);
    tensor_destroy(out3);
    tensor_destroy(out4);
    tensor_destroy(out5);

    x = NULL; 
    w = NULL;
    m = NULL;

    out1 = NULL; 
    out2 = NULL;
    out3 = NULL;
    out4 = NULL; 
    out5 = NULL;  
}

void take_step_SGD(float32_t learning_rate, float32_t momentum, float32_t dampening, float32_t weight_decay, bool_t nesterov, int steps)
{
    initialize_xwm();

    stochastic_gradient_descent_t *SGD = NULL;

    error = stochastic_gradient_descent_create(&SGD,
                                            FLOAT32,
                                            &learning_rate,
                                            &momentum,
                                            &dampening,
                                            &weight_decay,
                                            nesterov);

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

        if(!outNW)
        {
            error = ERROR(ERROR_CREATE, string_create("Failed to perform TinyNet forward using NW tensors."), error);
            destory();
            error_print(error);
            error_destroy(error); 
            return;

        }
        
        error = tensor_backward(outNW, NULL);
        if (error)
        {
            destory();
            error_print(error);
            error_destroy(error); 
            return;
        }

        error = stochastic_gradient_descent(SGD, x);
        if (error)
        {
            destory();
            error_print(error);
            error_destroy(error); 
            return;
        }
        error = stochastic_gradient_descent(SGD, w);
        if (error)
        {
            destory();
            error_print(error);
            error_destroy(error); 
            return;
        }
        outNW =NULL;
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

Suite *make_ptimizers_suite(void)
{
    Suite *s;
    TCase *tc_unary;

    s = suite_create("Test Optimizers Suite");

    tc_unary = tcase_create("Optimizers Case");
    tcase_add_test(tc_unary, test_SGD);
    tcase_add_test(tc_unary, test_sgd_high_lr);
    tcase_add_test(tc_unary, test_sgd_wd); //TODO not implemented yet
    tcase_add_test(tc_unary, test_sgd_high_lr_wd); //TODO not implemented yet
    tcase_add_test(tc_unary, test_multistep_sgd);
    tcase_add_test(tc_unary, test_multistep_sgd_high_lr);
    tcase_add_test(tc_unary, test_multistep_sgd_wd); //TODO not implemented yet
    tcase_add_test(tc_unary, test_multistep_sgd_high_lr_wd); //TODO not implemented yet

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