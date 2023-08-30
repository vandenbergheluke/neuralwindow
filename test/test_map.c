
#include <check.h>
#include <map.h>

map_t *map;
nw_error_t *error;

void setup(void)
{
    error = map_create(&map);
    if (error != NULL)
    {
       error_print(error);
       error_destroy(error); 
    }
}

void teardown(void)
{
    map_destroy(map);
}

START_TEST(test_map)
{
   string_t test_case_key_1 = string_create("1");
   string_t test_case_key_2 = string_create("2");
   string_t test_case_key_3 = string_create("3");
   int test_case_value_1 = 1;
   int test_case_value_2 = 2;
   int test_case_value_3 = 3;
   ck_assert(!map_contains(map, test_case_key_1));
   ck_assert(!map_contains(map, test_case_key_2));
   ck_assert(!map_contains(map, test_case_key_3));
   error = map_set(map, test_case_key_1, (void *) &test_case_value_1);
   if (error != NULL)
   {
      error_print(error);
      error_destroy(error); 
   }
   ck_assert_uint_eq(map->length, 1);
   ck_assert(map_contains(map, test_case_key_1));
   ck_assert(!map_contains(map, test_case_key_2));
   ck_assert(!map_contains(map, test_case_key_3));

   error = map_set(map, test_case_key_2, (void *) &test_case_value_2);
   if (error != NULL)
   {
      error_print(error);
      error_destroy(error); 
   }
   ck_assert_uint_eq(map->length, 2);
   ck_assert(map_contains(map, test_case_key_1));
   ck_assert(map_contains(map, test_case_key_2));
   ck_assert(!map_contains(map, test_case_key_3));

   error = map_set(map, test_case_key_3, (void *) &test_case_value_3);
   if (error != NULL)
   {
      error_print(error);
      error_destroy(error); 
   }
   ck_assert_uint_eq(map->length, 3);
   ck_assert(map_contains(map, test_case_key_1));
   ck_assert(map_contains(map, test_case_key_2));
   ck_assert(map_contains(map, test_case_key_3));

   int *test_case_value_ptr_1;
   int *test_case_value_ptr_2;
   int *test_case_value_ptr_3;

   error = map_get(map, test_case_key_1, (void *) &test_case_value_ptr_1);
   if (error != NULL)
   {
      error_print(error);
      error_destroy(error); 
   }
   ck_assert_int_eq(test_case_value_1, *test_case_value_ptr_1);

   error = map_get(map, test_case_key_2, (void *) &test_case_value_ptr_2);
   if (error != NULL)
   {
      error_print(error);
      error_destroy(error); 
   }
   ck_assert_int_eq(test_case_value_2, *test_case_value_ptr_2);

   error = map_get(map, test_case_key_3, (void *) &test_case_value_ptr_3);
   if (error != NULL)
   {
      error_print(error);
      error_destroy(error); 
   }
   ck_assert_int_eq(test_case_value_3, *test_case_value_ptr_3);
}
END_TEST

Suite *map_creation_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Test Suite");
    tc_core = tcase_create("Case 1");
    tcase_add_checked_fixture(tc_core, setup, teardown);
    tcase_add_test(tc_core, test_map);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    SRunner *sr;

    sr = srunner_create(map_creation_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
