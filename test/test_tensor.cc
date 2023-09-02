#include <iostream>
extern "C"
{
#include <check.h>
#include <view.h>
#include <buffer.h>
#include <tensor.h>
}
#include <torch/torch.h>

nw_error_t *error;

#define SEED 1234

bool_t set_seed = true;

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

    error = view_create(&view, 
                        (uint64_t) X.storage_offset(),
                        (uint64_t) X.ndimension(),
                        (uint64_t *) X.sizes().data(),
                        NULL);
    ck_assert_ptr_null(error);
    // error = buffer_create(&buffer[i],
    //                         runtimes[i],
    //                         datatypes[i],
    //                         unary_views[i],
    //                         (void *) unary_tensors[i].data_ptr(),
    //                         (uint64_t) unary_tensors[i].numel(),
    //                         true);
    // ck_assert_ptr_null(unary_error);
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
