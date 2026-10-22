/**
 * Copyright 2013-2023 Software Radio Systems Limited
 *
 * This file is part of srsRAN.
 */

#ifndef SRSRAN_CONFIG_H
#define SRSRAN_CONFIG_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <complex.h>

#if defined _WIN32 || defined __CYGWIN__
#define SRSRAN_IMPORT __declspec(dllimport)
#define SRSRAN_EXPORT __declspec(dllexport)
#define SRSRAN_LOCAL
#else
#define SRSRAN_IMPORT
#define SRSRAN_EXPORT
#define SRSRAN_LOCAL
#endif

#define SRSRAN_API SRSRAN_EXPORT

#define CONCAT(a, b) a##b
#define CONCAT2(a, b) CONCAT(a, b)
#define STRING2(x) #x
#define STRING(x) STRING2(x)

#define SRSRAN_SUCCESS 0
#define SRSRAN_ERROR -1
#define SRSRAN_ERROR_INVALID_INPUTS -2
#define SRSRAN_ERROR_TIMEOUT -3
#define SRSRAN_ERROR_INVALID_COMMAND -4
#define SRSRAN_ERROR_OUT_OF_BOUNDS -5
#define SRSRAN_ERROR_CANT_START -6
#define SRSRAN_ERROR_ALREADY_STARTED -7
#define SRSRAN_ERROR_RX_EOF -8

#if defined(_MSC_VER)
typedef _Fcomplex cf_t;
#else
typedef _Complex float cf_t;
#endif

#endif // SRSRAN_CONFIG_H
