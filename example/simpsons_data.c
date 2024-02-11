#include <simpsons_data.h>
#include <tensor.h>

nw_error_t *simpsons_setup(void *arguments) 
{
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    nw_error_t *error = NULL;
    int character;
    simpsons_dataset_t *simpsons_dataset = (simpsons_dataset_t *) arguments;
    simpsons_dataset->vocabulary_size = 0;
    simpsons_dataset->number_of_characters = 0;

    for (int i = 0; i < CHAR_MAX; ++i)
    {
        simpsons_dataset->integer_to_character[i] = '\0';
        simpsons_dataset->character_to_integer[i] = -1;
    }

    simpsons_dataset->data_file = fopen(simpsons_dataset->data_path, "r");
    if (!simpsons_dataset->data_file)
    {
        return ERROR(ERROR_FILE, string_create("failed to open %s.", simpsons_dataset->data_path), NULL);
    }

    while ((character = fgetc(simpsons_dataset->data_file)) != EOF)
    {
        if (simpsons_dataset->character_to_integer[character] == -1)
        {
            simpsons_dataset->character_to_integer[character] = simpsons_dataset->vocabulary_size;
            simpsons_dataset->integer_to_character[simpsons_dataset->vocabulary_size] = character;
            ++simpsons_dataset->vocabulary_size;
        }
        ++simpsons_dataset->number_of_characters;
    }

    return error;
}

nw_error_t *simpsons_teardown(void *arguments) 
{
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    int status;
    simpsons_dataset_t *simpsons_dataset = (simpsons_dataset_t *) arguments;

    status = fclose(simpsons_dataset->data_file);
    if (status)
    {
        return ERROR(ERROR_FILE, string_create("failed to close file %s.", simpsons_dataset->data_file), NULL);
    }

    return NULL;
}

nw_error_t *simpsons_dataloader(int64_t index, batch_t *batch, void *arguments)
{
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(batch, "batch");

    int status;
    nw_error_t *error = NULL;
    simpsons_dataset_t *simpsons_dataset = (simpsons_dataset_t *) arguments;
    int64_t batch_size = batch->batch_size;
    int64_t block_size = simpsons_dataset->block_size;
    void *data = NULL;
    void *labels = NULL;
    datatype_t datatype = batch->datatype;
    runtime_t runtime = batch->runtime;
    int previous_character, next_character;
    bool_t copy = runtime == CU_RUNTIME;
    size_t size = datatype_size(datatype) * batch_size * block_size;

    data = (void *) malloc(size);
    if (!data)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
    }

    labels = (void *) malloc(size);
    if (!labels)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
    }

    status = fseek(simpsons_dataset->data_file, index * block_size, SEEK_SET);
    if (status)
    {
        return ERROR(ERROR_FILE, string_create("failed to move to offset in file."), NULL);
    }

    previous_character = fgetc(simpsons_dataset->data_file);
    if (previous_character == EOF)
    {
        return ERROR(ERROR_FILE, string_create("reached end of file."), NULL);
    }

    for (int64_t i = 0; i < batch_size * block_size; ++i)
    {
        next_character = fgetc(simpsons_dataset->data_file);
        if (next_character == EOF)
        {
            return ERROR(ERROR_FILE, string_create("reached end of file."), NULL);
        }

        switch (datatype)
        {
        case FLOAT32:
            ((float32_t *) data)[i] = (float32_t) simpsons_dataset->character_to_integer[previous_character];
            ((float32_t *) labels)[i] = (float32_t) simpsons_dataset->character_to_integer[next_character];
            break;
        case FLOAT64:
            ((float64_t *) data)[i] = (float64_t) simpsons_dataset->character_to_integer[previous_character];
            ((float64_t *) labels)[i] = (float64_t) simpsons_dataset->character_to_integer[next_character];
            break;
        default:
            return ERROR(ERROR_DATATYPE, string_create("unsupported datatype."), NULL);
        }
        previous_character = next_character;
    }

    error = tensor_from_data(&batch->x, data, runtime, datatype, 2, (int64_t[]) {batch_size, block_size}, copy, false, true);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    error = tensor_from_data(&batch->y, labels, runtime, datatype, 2, (int64_t[]) {batch_size * block_size, 1}, copy, false, true);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    if (copy)
    {
        free(data);
        free(labels);
    }

    return error;
}

