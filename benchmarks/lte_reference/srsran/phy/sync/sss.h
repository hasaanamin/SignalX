/* Test-only declarations used by gen_sss.c. No sequence/correlation algorithm. */
#include <stdint.h>
#include <stdbool.h>
#define SRSRAN_SSS_N 31
typedef struct {
    int s[31][31];
    int z1[31][31];
    int c[2][31];
} srsran_sss_tables_t;
