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
#include <layer.h>
#include <init.h>
#include <test_helper.h>
}
#include <test_helper_torch.h>

#define CASES 3
nw_error_t *error = NULL;
model_t *models[RUNTIMES][DATATYPES][CASES];

void setup_single_layer_feed_forward(runtime_t runtime, datatype_t datatype, model_t **model)
{
    // NW Model
    layer_t *hidden_layer = NULL;
    layer_t *output_layer = NULL;
    layer_t *hidden_activation_layer = NULL;
    layer_t *output_activation_layer = NULL;
    block_t *block = NULL;
    parameter_init_t *weight_init = NULL;
    parameter_init_t *bias_init = NULL;
    void *mean = NULL;
    void *standard_deviation = NULL;

    mean = (void *) malloc(datatype_size(datatype));
    ck_assert_ptr_nonnull(mean);
    standard_deviation = (void *) malloc(datatype_size(datatype));
    ck_assert_ptr_nonnull(standard_deviation);

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) mean = (float32_t) 0.0;
        *(float32_t *) standard_deviation = (float32_t) 0.2;
        break;
    case FLOAT64:
        *(float64_t *) mean = (float64_t) 0.0;
        *(float64_t *) standard_deviation = (float64_t) 0.2;
        break;
    default:
        ck_abort_msg("unknown datatype");
    }

    error = normal_parameter_init(&weight_init, mean, standard_deviation, datatype);
    ck_assert_ptr_null(error);

    error = normal_parameter_init(&bias_init, mean, standard_deviation, datatype);
    ck_assert_ptr_null(error);

    error = rectified_linear_activation_layer_create(&hidden_activation_layer);
    ck_assert_ptr_null(error);
    ck_assert_ptr_nonnull(hidden_activation_layer);

    error = softmax_activation_layer_create(&output_activation_layer, -1);
    ck_assert_ptr_null(error);
    ck_assert_ptr_nonnull(output_activation_layer);

    error = linear_layer_create(&hidden_layer, 10, 20, runtime, datatype, weight_init, bias_init);
    ck_assert_ptr_null(error);
    ck_assert_ptr_nonnull(hidden_layer);

    error = linear_layer_create(&output_layer, 20, 10, runtime, datatype, weight_init, bias_init);
    ck_assert_ptr_null(error);
    ck_assert_ptr_nonnull(output_layer);

    error = block_create(&block, 4, hidden_layer, hidden_activation_layer, output_layer, output_activation_layer);
    ck_assert_ptr_null(error);
    ck_assert_ptr_nonnull(block);

    error = model_create(model, block);
    ck_assert_ptr_null(error);

    parameter_init_destroy(weight_init);
    parameter_init_destroy(bias_init);
    free(mean);
    free(standard_deviation);
}

void setup_convolutional_neural_network(runtime_t runtime, datatype_t datatype, model_t **model)
{
    // NW Model
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
    void *mean = NULL;
    void *standard_deviation = NULL;
    parameter_init_t *weight_init = NULL;
    parameter_init_t *bias_init = NULL;

    momentum = (void *) malloc(datatype_size(datatype));
    ck_assert_ptr_nonnull(momentum);
    epsilon = (void *) malloc(datatype_size(datatype));
    ck_assert_ptr_nonnull(epsilon);
    c = (void *) malloc(datatype_size(datatype));
    ck_assert_ptr_nonnull(c);
    mean = (void *) malloc(datatype_size(datatype));
    ck_assert_ptr_nonnull(mean);
    standard_deviation = (void *) malloc(datatype_size(datatype));
    ck_assert_ptr_nonnull(standard_deviation);

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) momentum = (float32_t) 0.1;
        *(float32_t *) epsilon = (float32_t) 1e-5;
        *(float32_t *) c = (float32_t) 0.2;
        *(float32_t *) mean = (float32_t) 0.0;
        *(float32_t *) standard_deviation = (float32_t) 0.2;
        break;
    case FLOAT64:
        *(float64_t *) momentum = (float64_t) 0.1;
        *(float64_t *) epsilon = (float64_t) 1e-5;
        *(float64_t *) c = (float64_t) 0.2;
        *(float64_t *) mean = (float64_t) 0.0;
        *(float64_t *) standard_deviation = (float64_t) 0.2;
        break;
    default:
        ck_abort_msg("unknown datatype");
    }

    error = normal_parameter_init(&weight_init, mean, standard_deviation, datatype);
    ck_assert_ptr_null(error);

    error = normal_parameter_init(&bias_init, mean, standard_deviation, datatype);
    ck_assert_ptr_null(error);

    error = convolution_transpose_2d_layer_create(&convtranspose1, 4, 0, 1, 5, 2, runtime, datatype, weight_init, bias_init);
    ck_assert_ptr_null(error);

    error = convolution_transpose_2d_layer_create(&convtranspose2, 3, 1, 2, 2, 3, runtime, datatype, weight_init, bias_init);
    ck_assert_ptr_null(error);

    error = convolution_transpose_2d_layer_create(&convtranspose3, 4, 1, 2, 3, 4, runtime, datatype, weight_init, bias_init);
    ck_assert_ptr_null(error);

    error = convolution_transpose_2d_layer_create(&convtranspose4, 3, 1, 2, 4, 1, runtime, datatype, weight_init, NULL);
    ck_assert_ptr_null(error);

    error = convolution_2d_layer_create(&conv1, 4, 1, 2, 1, 2, runtime, datatype, weight_init, bias_init);
    ck_assert_ptr_null(error);

    error = convolution_2d_layer_create(&conv2, 4, 1, 2, 2, 3, runtime, datatype, weight_init, NULL);
    ck_assert_ptr_null(error);

    error = convolution_2d_layer_create(&conv3, 4, 1, 2, 3, 4, runtime, datatype, weight_init, bias_init);
    ck_assert_ptr_null(error);

    error = convolution_2d_layer_create(&conv4, 3, 0, 1, 4, 1, runtime, datatype, weight_init, bias_init);
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

    error = model_create(model, block);
    ck_assert_ptr_null(error);

    parameter_init_destroy(weight_init);
    parameter_init_destroy(bias_init);
    free(momentum);
    free(epsilon);
    free(c);
    free(mean);
    free(standard_deviation);
}

void setup_transformer(runtime_t runtime, datatype_t datatype, model_t **model)
{
    // Parameters
    int64_t vocabulary_size = 5;
    int64_t embedding_size = 4;
    int64_t number_of_heads = 2;
    int64_t block_size = 3;
    void *epsilon = NULL;
    void *dropout_probability = NULL;
    void *mean = NULL;
    void *standard_deviation = NULL;
    parameter_init_t *weight_init = NULL;
    parameter_init_t *bias_init = NULL;

    epsilon = (void *) malloc(datatype_size(datatype));
    ck_assert_ptr_nonnull(epsilon);
    dropout_probability = (void *) malloc(datatype_size(datatype));
    ck_assert_ptr_nonnull(dropout_probability);
    mean = (void *) malloc(datatype_size(datatype));
    ck_assert_ptr_nonnull(mean);
    standard_deviation = (void *) malloc(datatype_size(datatype));
    ck_assert_ptr_nonnull(standard_deviation);

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) epsilon = (float32_t) 1e-5;
        *(float32_t *) dropout_probability = (float32_t) 0.1;
        *(float32_t *) mean = (float32_t) 0.0;
        *(float32_t *) standard_deviation = (float32_t) 0.2;
        break;
    case FLOAT64:
        *(float64_t *) epsilon = (float64_t) 1e-5;
        *(float64_t *) dropout_probability = (float64_t) 0.1;
        *(float64_t *) mean = (float64_t) 0.0;
        *(float64_t *) standard_deviation = (float64_t) 0.2;
        break;
    default:
        ck_abort_msg("unknown datatype");
    }

    error = normal_parameter_init(&weight_init, mean, standard_deviation, datatype);
    ck_assert_ptr_null(error);

    error = normal_parameter_init(&bias_init, mean, standard_deviation, datatype);
    ck_assert_ptr_null(error);

    // NW Model
    int64_t normal_shape[] = {embedding_size};
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

    error = layer_normalization_layer_create(&layer_norm_1, normal_shape, 1, epsilon, true, true, datatype, runtime);
    ck_assert_ptr_null(error);

    error = layer_normalization_layer_create(&layer_norm_2, normal_shape, 1, epsilon, true, true, datatype, runtime);
    ck_assert_ptr_null(error);

    error = layer_normalization_layer_create(&layer_norm_3, normal_shape, 1, epsilon, true, true, datatype, runtime);
    ck_assert_ptr_null(error);

    error = causal_multihead_self_attention_layer_create(&causal_multihead_self_attention, number_of_heads, embedding_size, dropout_probability, datatype, runtime,
                                                         weight_init, bias_init, weight_init, NULL);
    ck_assert_ptr_null(error);

    error = linear_layer_create(&linear_1, embedding_size, 4 * embedding_size, runtime, datatype, weight_init, bias_init);
    ck_assert_ptr_null(error);

    error = linear_layer_create(&linear_2, 4 * embedding_size, embedding_size, runtime, datatype, weight_init, bias_init);
    ck_assert_ptr_null(error);

    error = linear_layer_create(&linear_3, embedding_size, vocabulary_size, runtime, datatype, weight_init, NULL);
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

    error = transformer_embedding_layer_create(&transformer_embedding, vocabulary_size, embedding_size, block_size, datatype, runtime, weight_init, weight_init);
    ck_assert_ptr_null(error);

    error = block_create(&block, 6, transformer_embedding, dropout_1, decoder, layer_norm_3, linear_3, reshape);
    ck_assert_ptr_null(error);

    error = model_create(model, block);
    ck_assert_ptr_null(error);

    free(epsilon);
    free(dropout_probability);
    parameter_init_destroy(weight_init);
    parameter_init_destroy(bias_init);
    free(mean);
    free(standard_deviation);
}

void setup_model(runtime_t runtime, datatype_t datatype, int case_index)
{
    switch (case_index)
    {
    case 0:
        setup_single_layer_feed_forward(runtime, datatype, &models[runtime][datatype][case_index]);
        break;
    case 1:
        setup_convolutional_neural_network(runtime, datatype, &models[runtime][datatype][case_index]);
        break;
    case 2:
        setup_transformer(runtime, datatype, &models[runtime][datatype][case_index]);
        break;
    default:
        ck_abort_msg("unsupported case.");
    }
}

void setup(void)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        runtime_t runtime = (runtime_t) i;
        runtime_create_context(runtime);
        for (int j = 0; j < DATATYPES; ++j)
        {
            datatype_t datatype = (datatype_t) j;
            for (int k = 0; k < CASES; ++k)
            {
                models[i][j][k] = NULL;
                setup_model(runtime, datatype, k);
            }
        }
    }
}

void teardown(void)
{
    error_print(error);
    error_destroy(error);
    for (int i = 0; i < RUNTIMES; ++i)
    {
        runtime_t runtime = (runtime_t) i;
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                model_destroy(models[i][j][k]);
                models[i][j][k] = NULL;
            }
        }
        runtime_destroy_context(runtime);
    }
}

START_TEST(test_model_exporter)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                model_t *returned_model = NULL;
                error = model_save(models[i][j][k], "test.bin");
                ck_assert_ptr_null(error);
                error = model_load(&returned_model, "test.bin");
                ck_assert_ptr_null(error);
                ck_assert_model_eq(returned_model, models[i][j][k]);
                model_destroy(returned_model);
            }
        }
    }
}
END_TEST

Suite *make_model_exporter_suite(void)
{
    Suite *s;
    TCase *tc;

    s = suite_create("Test Model Exporter Suite");

    tc = tcase_create("Test Model Exporter Case");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, test_model_exporter);

    suite_add_tcase(s, tc);

    return s;
}

int main(void) 
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_model_exporter_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}