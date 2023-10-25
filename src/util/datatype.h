/**@file datatype.h
 * @brief Provides common datatypes and utilities that interact with them.
 *
 */

#ifndef DATATYPE_H
#define DATATYPE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>

typedef float float32_t;
typedef double float64_t;
typedef bool bool_t;
typedef const char * string_t;
typedef char char_t;
typedef unsigned char uchar_t;

#ifdef __cplusplus
typedef enum datatype_t: int
#else
typedef enum datatype_t
#endif
{
    FLOAT32,
    FLOAT64
} datatype_t;

// >=C++14
// Defines that force the compiler to choose the correct overloaded function
// when passing it as an argument.
//
// AS_MEMBER_LAMBDA has the additional functionality of taking its first
// argument as the object whose member is to be called.
#if defined (__cplusplus) && (__cplusplus >= 201402L)
#define AS_LAMBDA(func) [&](auto&&... args) -> decltype(func(std::forward<decltype(args)>(args)...)) { return func(std::forward<decltype(args)>(args)...); }
#define AS_MEMBER_LAMBDA(func) [&](auto obj, auto&&... args) -> decltype(obj.func(std::forward<decltype(args)>(args)...)) { return obj.func(std::forward<decltype(args)>(args)...); }
#endif

#define DATATYPES 2

string_t datatype_string(datatype_t datatype);
size_t datatype_size(datatype_t datatype);
string_t string_create(string_t format, ...);
void string_destroy(string_t string);

#endif
