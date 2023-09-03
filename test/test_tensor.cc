#include "datatype.h"
#include <ATen/core/TensorBody.h>
#include <ATen/ops/sum.h>
#include <iostream>
extern "C"
{
#include <check.h>
#include <view.h>
#include <buffer.h>
#include <tensor.h>
}
#include <torch/torch.h>
#include <cstring>

nw_error_t *error;

#define SEED 1234

bool_t set_seed = true;

runtime_t runtimes[] = {
   OPENBLAS_RUNTIME,
   MKL_RUNTIME,
   CU_RUNTIME
};

void setup(void)
{
    if (set_seed)
    {
        torch::manual_seed(SEED);
        set_seed = false;
    }

    error = NULL;
}

void teardown(void)
{
    error_destroy(error);
}

START_TEST(test_exponential)
{
    torch::Tensor X = torch::randn({3, 3}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
    torch::Tensor Y = torch::exp(X);
    torch::Tensor J = torch::sum(Y);
    J.backward();

    view_t *view;
    buffer_t *buffer;
    tensor_t *nw_X;
    tensor_t *nw_Y;
    tensor_t *nw_J;

    for (long unsigned int i = 0; i < (sizeof(runtimes) / sizeof(runtimes[0])); i++) {
        error = view_create(&view, 
                            (uint64_t) X.storage_offset(),
                            (uint64_t) X.ndimension(),
                            (uint64_t *) X.sizes().data(),
                            NULL);
        ck_assert_ptr_null(error);
        error = buffer_create(&buffer,
                                runtimes[i],
                                FLOAT32,
                                view,
                                (void *) X.data_ptr(),
                                (uint64_t) X.numel(),
                                true);
        ck_assert_ptr_null(error);
        error = tensor_create(&nw_X,
                                buffer,
                                NULL,
                                NULL,
                                true,
                                false);
        ck_assert_ptr_null(error);

        error = tensor_create_empty(&nw_Y);
        ck_assert_ptr_null(error);

        error = tensor_create_empty(&nw_J);
        ck_assert_ptr_null(error);

        error = tensor_exponential(nw_X, nw_Y);
        ck_assert_ptr_null(error);

        error = tensor_summation(nw_Y,
                                    nw_J,
                                    (uint64_t *) X.sizes().data(),
                                    (uint64_t) X.ndimension(),
                                    false);
        ck_assert_ptr_null(error);

        error = tensor_backward(nw_J, NULL);
        ck_assert_ptr_null(error);

        error = tensor_accumulate_gradient(nw_X, nw_J->gradient); 
        ck_assert_ptr_null(error);

        ck_assert(std::memcmp(nw_X->gradient->buffer->data,
                                X.grad().data_ptr(),
                                X.grad().numel() * sizeof(float32_t)) == 0);

        tensor_destroy(nw_X);
        tensor_destroy(nw_Y);
        tensor_destroy(nw_J);
    }
}
END_TEST

Suite *make_sample_creation_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Test Automatic Differentiation Suite");
    tc_core = tcase_create("Test Tensor Case");
    tcase_add_checked_fixture(tc_core, setup, teardown);
    tcase_add_test(tc_core, test_exponential);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_sample_creation_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
