/* srsran_fec_adapter.c: C ABI adapter exposing srsRAN Turbo, Rate Matching, and CRC */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "srsran/config.h"
#include "srsran/srsran.h"
#include "srsran/phy/fec/crc.h"
#include "srsran/phy/fec/cbsegm.h"
#include "srsran/phy/fec/turbo/tc_interl.h"
#include "srsran/phy/fec/turbo/turbocoder.h"
#include "srsran/phy/fec/turbo/turbodecoder.h"
#include "srsran/phy/fec/turbo/rm_turbo.h"
#include "srsran/phy/utils/bit.h"
#include "srsran/phy/utils/vector.h"

#define CRC24A_POLY 0x1864cfb
#define CRC24B_POLY 0x1800063

/* C symbol for bzero if called by object files */
void bzero(void *s, size_t n) {
    memset(s, 0, n);
}

/* Vector and memory helpers for srsRAN */
void* srsran_vec_malloc(uint32_t size) {
    return malloc(size);
}
int16_t* srsran_vec_i16_malloc(uint32_t size) {
    return (int16_t*)malloc(size * sizeof(int16_t));
}
uint16_t* srsran_vec_u16_malloc(uint32_t size) {
    return (uint16_t*)malloc(size * sizeof(uint16_t));
}
uint8_t* srsran_vec_u8_malloc(uint32_t size) {
    return (uint8_t*)malloc(size * sizeof(uint8_t));
}
void srsran_vec_sub_sss(const int16_t* x, const int16_t* y, int16_t* z, const uint32_t len) {
    for (uint32_t i = 0; i < len; i++) {
        z[i] = x[i] - y[i];
    }
}
void srsran_vec_sub_bbb(const int8_t* x, const int8_t* y, int8_t* z, const uint32_t len) {
    for (uint32_t i = 0; i < len; i++) {
        z[i] = x[i] - y[i];
    }
}
void srsran_vec_lut_sss(const short* x, const unsigned short* lut, short* y, const uint32_t len) {
    for (uint32_t i = 0; i < len; i++) {
        y[lut[i]] = x[i];
    }
}
void srsran_vec_lut_bbb(const int8_t* x, const unsigned short* lut, int8_t* y, const uint32_t len) {
    for (uint32_t i = 0; i < len; i++) {
        y[lut[i]] = x[i];
    }
}

int srsran_tc_interl_init(srsran_tc_interl_t* h, uint32_t max_long_cb) {
    h->forward = (uint16_t*)malloc(max_long_cb * sizeof(uint16_t));
    h->reverse = (uint16_t*)malloc(max_long_cb * sizeof(uint16_t));
    if (!h->forward || !h->reverse) {
        if (h->forward) free(h->forward);
        if (h->reverse) free(h->reverse);
        return -1;
    }
    h->max_long_cb = max_long_cb;
    return 0;
}

void srsran_tc_interl_free(srsran_tc_interl_t* h) {
    if (h->forward) free(h->forward);
    if (h->reverse) free(h->reverse);
    memset(h, 0, sizeof(srsran_tc_interl_t));
}

/* Exported functions */
#define EXPORT __declspec(dllexport)

EXPORT int srsran_fec_crc24a_checksum(const uint8_t *data_bits, uint32_t len_bits) {
    srsran_crc_t crc;
    srsran_crc_init(&crc, CRC24A_POLY, 24);
    /* srsran_crc_checksum operates on bit array */
    return (int)srsran_crc_checksum(&crc, (uint8_t*)data_bits, (int)len_bits);
}

EXPORT int srsran_fec_crc24a_verify(const uint8_t *data_bits_with_crc, uint32_t total_len_bits) {
    if (total_len_bits < 24) return 0;
    srsran_crc_t crc;
    srsran_crc_init(&crc, CRC24A_POLY, 24);
    uint32_t data_len = total_len_bits - 24;
    uint32_t calc = srsran_crc_checksum(&crc, (uint8_t*)data_bits_with_crc, (int)data_len);
    for (uint32_t i = 0; i < 24; i++) {
        uint8_t expected_bit = (calc >> (23 - i)) & 1;
        if (data_bits_with_crc[data_len + i] != expected_bit) {
            return 0;
        }
    }
    return 1;
}

EXPORT int srsran_fec_turbo_encode(const uint8_t *in_bits, uint8_t *out_bits, uint32_t K) {
    srsran_tcod_t tcod;
    if (srsran_tcod_init(&tcod, K) != SRSRAN_SUCCESS) {
        return -1;
    }
    srsran_tcod_encode(&tcod, (uint8_t*)in_bits, out_bits, K);
    srsran_tcod_free(&tcod);
    return 0;
}

EXPORT int srsran_fec_turbo_decode(const float *llr, uint32_t coded_len, uint8_t *out_bits, uint32_t K, uint32_t max_iterations) {
    srsran_tdec_t tdec;
    if (srsran_tdec_init_manual(&tdec, K, SRSRAN_TDEC_GENERIC) != SRSRAN_SUCCESS) {
        return -1;
    }
    srsran_tdec_force_not_sb(&tdec);
    srsran_tdec_new_cb(&tdec, K);

    int16_t *llr_s = (int16_t*)malloc(coded_len * sizeof(int16_t));
    if (!llr_s) {
        srsran_tdec_free(&tdec);
        return -2;
    }
    for (uint32_t i = 0; i < coded_len; i++) {
        float val = llr[i] * 16.0f;
        if (val > 32767.0f) val = 32767.0f;
        if (val < -32768.0f) val = -32768.0f;
        llr_s[i] = (int16_t)val;
    }

    uint8_t *data_bytes = (uint8_t*)malloc((K + 7) / 8);
    if (!data_bytes) {
        free(llr_s);
        srsran_tdec_free(&tdec);
        return -3;
    }

    int iters = srsran_tdec_run_all(&tdec, llr_s, data_bytes, max_iterations, K);

    srsran_bit_unpack_vector(data_bytes, out_bits, K);

    free(data_bytes);
    free(llr_s);
    srsran_tdec_free(&tdec);
    return iters;
}

EXPORT int srsran_fec_rm_turbo_rx(const float *in_llr, uint32_t in_len, float *out_llr, uint32_t out_len, uint32_t rv_idx, uint32_t filler_bits) {
    uint32_t nrows = (uint32_t)(out_len / 3 - 1) / 32 + 1;
    uint32_t K_p = nrows * 32;
    uint32_t buff_len = 3 * K_p;
    float *w_buff = (float*)malloc(buff_len * sizeof(float));
    if (!w_buff) return -1;
    for (uint32_t i = 0; i < buff_len; i++) w_buff[i] = (float)SRSRAN_RX_NULL;

    int res = srsran_rm_turbo_rx(w_buff, buff_len, (float*)in_llr, in_len, out_llr, out_len, rv_idx, filler_bits);
    free(w_buff);
    return res;
}

EXPORT int srsran_fec_rm_turbo_tx(const uint8_t *in_bits, uint32_t in_len, uint8_t *out_bits, uint32_t out_len, uint32_t rv_idx) {
    uint32_t nrows = (uint32_t)(in_len / 3 - 1) / 32 + 1;
    uint32_t K_p = nrows * 32;
    uint32_t buff_len = 3 * K_p;
    uint8_t *w_buff = (uint8_t*)malloc(buff_len);
    if (!w_buff) return -1;
    memset(w_buff, (int)SRSRAN_TX_NULL, buff_len);

    int res = srsran_rm_turbo_tx(w_buff, buff_len, (uint8_t*)in_bits, in_len, out_bits, out_len, rv_idx);
    free(w_buff);
    return res;
}

/* Complete transport block decode: rate unmatching -> turbo decode -> CRC verification */
EXPORT int srsran_fec_decode_tb(const float *rx_llr, uint32_t E, uint8_t *out_payload_bits, uint32_t tbs, uint32_t rv_idx, uint32_t max_iterations, int *crc_passed) {
    uint32_t K = tbs + 24;
    uint32_t coded_len = 3 * K + 12;

    float *unmatched_llr = (float*)malloc(coded_len * sizeof(float));
    if (!unmatched_llr) return -1;

    int rm_res = srsran_fec_rm_turbo_rx(rx_llr, E, unmatched_llr, coded_len, rv_idx, 0);
    if (rm_res != SRSRAN_SUCCESS) {
        free(unmatched_llr);
        return -2;
    }

    uint8_t *decoded_bits = (uint8_t*)malloc(K);
    if (!decoded_bits) {
        free(unmatched_llr);
        return -3;
    }

    int iters = srsran_fec_turbo_decode(unmatched_llr, coded_len, decoded_bits, K, max_iterations);
    free(unmatched_llr);

    if (iters < 0) {
        free(decoded_bits);
        return -4;
    }

    *crc_passed = srsran_fec_crc24a_verify(decoded_bits, K);

    memcpy(out_payload_bits, decoded_bits, tbs);
    free(decoded_bits);

    return iters;
}
