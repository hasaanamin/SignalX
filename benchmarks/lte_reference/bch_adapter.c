/* Test-only ABI adapter; LTE coding functions are unchanged downloaded sources.
 * ABI headers below provide constants/structs only, not replacement algorithms. */
#include <stdint.h>
uint8_t Partab[256];
uint32_t P_init = 0;
void partab_init(void) {
    for (unsigned i=0; i<256; ++i) {
        unsigned parity_value=0;
        for (unsigned v=i; v; v>>=1) parity_value ^= v&1;
        Partab[i]=(uint8_t)parity_value;
    }
    P_init=1;
}
#include "convcoder.c"
#include "rm_conv.c"
__declspec(dllexport) void reference_bch(uint8_t *bits, uint8_t *encoded, uint8_t *matched) {
    srsran_convcoder_t encoder = {7, 3, {0x6d, 0x4f, 0x57}, 1};
    srsran_convcoder_encode(&encoder, bits, encoded, 40);
    srsran_rm_conv_tx(encoded, 120, matched, 1920);
}
