/* Test ABI only, matching pinned srsRAN convcoder.h. */
#include <stdint.h>
#define SRSRAN_ERROR_INVALID_INPUTS -2
typedef struct { int K; int R; int poly[3]; int tail_biting; } srsran_convcoder_t;
