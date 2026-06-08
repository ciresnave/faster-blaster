/**
 * @file test_judge_gemm3m_regression.c
 * @brief Focused Judge regression for the complex GEMM3M alias/plain boundary.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#include <unistd.h>
#endif

#include "../include/faster-blaster/judge.h"
#include "../include/faster-blaster/backend_ids.h"
#include "../include/faster-blaster/backend_plugin.h"
#include "../src/backends/reference.h"
#include "../src/judge/judge_op_ids.h"
#include "../src/judge/judge_types.h"

#ifndef FB_REFERENCE_DLL_DIR
#define FB_REFERENCE_DLL_DIR "../faster-blaster-reference/build-extended"
#endif

extern void fb_judge_register_oracle(const fb_backend_vtable_t *vtable);
extern void fb_judge_register_backend(uint32_t backend_id,
                                      const fb_backend_vtable_t *vtable);

typedef struct {
    uint32_t op_id;
    uint8_t dtype;
    uint8_t min_digits;
    const char *name;
} fb_gemm3m_regression_case_t;

static const fb_gemm3m_regression_case_t k_cases[] = {
    { FB_OP_CBLAS_CGEMM3M_BATCH, FB_DTYPE_CF32, 5u, "cblas_cgemm3m_batch" },
    { FB_OP_CBLAS_CGEMM3M_BATCH_STRIDED, FB_DTYPE_CF32, 5u, "cblas_cgemm3m_batch_strided" },
    { FB_OP_CBLAS_ZGEMM3M_BATCH, FB_DTYPE_CF64, 12u, "cblas_zgemm3m_batch" },
    { FB_OP_CBLAS_ZGEMM3M_BATCH_STRIDED, FB_DTYPE_CF64, 12u, "cblas_zgemm3m_batch_strided" },
    { FB_OP_CGEMM3M_BATCH, FB_DTYPE_CF32, 5u, "cgemm3m_batch" },
    { FB_OP_ZGEMM3M_BATCH, FB_DTYPE_CF64, 12u, "zgemm3m_batch" },
    { FB_OP_CGEMM3M_BATCH_STRIDED, FB_DTYPE_CF32, 5u, "cgemm3m_batch_strided" },
    { FB_OP_ZGEMM3M_BATCH_STRIDED, FB_DTYPE_CF64, 12u, "zgemm3m_batch_strided" },
};

static const char *judge_status_str(fb_judge_status_t status)
{
    switch (status) {
        case FB_JUDGE_OK:
            return "OK";
        case FB_JUDGE_ERR_NOT_INITIALIZED:
            return "NOT_INITIALIZED";
        case FB_JUDGE_ERR_INVALID_OP:
            return "INVALID_OP";
        case FB_JUDGE_ERR_NO_PROFILE:
            return "NO_PROFILE";
        case FB_JUDGE_ERR_ORACLE_FAILURE:
            return "ORACLE_FAILURE";
        case FB_JUDGE_ERR_ALLOC:
            return "ALLOC_ERR";
        case FB_JUDGE_ERR_IO:
            return "IO_ERR";
        case FB_JUDGE_ERR_NOT_IMPL:
            return "NOT_IMPL";
        default:
            return "UNKNOWN";
    }
}

static unsigned long long judge_profile_nonce(void)
{
#if defined(_WIN32)
    return (((unsigned long long)GetCurrentProcessId()) << 32) ^
           (unsigned long long)GetTickCount64();
#else
    return (((unsigned long long)getpid()) << 32) ^
           (unsigned long long)time(NULL);
#endif
}

static void build_profile_dir(char *buf, size_t buf_size)
{
    const char *override = getenv("FB_JUDGE_PROFILE_DIR");
    const char *tmp = getenv("TEMP");

    if (!buf || buf_size == 0) {
        return;
    }

    if (override && override[0] != '\0') {
        snprintf(buf, buf_size, "%s", override);
        return;
    }

    if (!tmp) {
        tmp = getenv("TMPDIR");
    }
    if (!tmp) {
        tmp = "/tmp";
    }

    snprintf(buf, buf_size, "%s/fb_judge_gemm3m_regression_%llu",
             tmp, judge_profile_nonce());
}

static fb_lib_handle_t load_reference_library(void)
{
#if defined(_WIN32)
    const char *ref_names[] = {
        "faster_blaster_reference.dll",
        "libfaster_blaster_reference.dll",
        NULL
    };
#elif defined(__APPLE__)
    const char *ref_names[] = { "libfaster_blaster_reference.dylib", NULL };
#else
    const char *ref_names[] = { "libfaster_blaster_reference.so", NULL };
#endif
    const char *ref_paths[] = {
        FB_REFERENCE_DLL_DIR,
        ".",
        NULL
    };

    return fb_plugin_load_library(ref_names, ref_paths);
}

static int run_case(const fb_gemm3m_regression_case_t *test_case)
{
    fb_precision_profile_t profile;
    fb_judge_status_t status;

    if (!test_case) {
        fprintf(stderr, "[FAIL] Invalid regression case pointer\n");
        return 1;
    }

    memset(&profile, 0, sizeof(profile));
    status = fb_judge_run(test_case->op_id,
                          FB_BACKEND_ID_REFERENCE,
                          0u,
                          FB_SIZE_SMALL,
                          test_case->dtype,
                          false,
                          &profile);
    if (status != FB_JUDGE_OK) {
        fprintf(stderr, "[FAIL] %s returned %s (%d)\n",
                test_case->name,
                judge_status_str(status),
                (int)status);
        return 1;
    }

    if (profile.direct.test_case_count == 0) {
        fprintf(stderr, "[FAIL] %s produced zero direct test cases\n",
                test_case->name);
        return 1;
    }

    if (profile.direct.has_fatal_failure) {
        fprintf(stderr, "[FAIL] %s reported a fatal direct-profile failure\n",
                test_case->name);
        return 1;
    }

    if (profile.direct.guaranteed_digits < test_case->min_digits) {
        fprintf(stderr,
                "[FAIL] %s regressed to %u guaranteed digits (need >= %u)\n",
                test_case->name,
                (unsigned)profile.direct.guaranteed_digits,
                (unsigned)test_case->min_digits);
        return 1;
    }

    printf("[PASS] %-28s digits=%u p50=%u ns\n",
           test_case->name,
           (unsigned)profile.direct.guaranteed_digits,
           (unsigned)profile.timing.p50_ns);
    return 0;
}

int main(void)
{
    char profile_dir[512];
    fb_lib_handle_t ref_h = NULL;
    const fb_backend_vtable_t *reference_vtable = NULL;
    fb_judge_status_t init_status;
    size_t i;

    build_profile_dir(profile_dir, sizeof(profile_dir));
    init_status = fb_judge_init(profile_dir);
    if (init_status != FB_JUDGE_OK) {
        fprintf(stderr, "[FATAL] fb_judge_init('%s') failed: %s\n",
                profile_dir,
                judge_status_str(init_status));
        return 1;
    }

    ref_h = load_reference_library();
    if (!ref_h) {
        fprintf(stderr,
                "[FATAL] Reference DLL not found. Tried '%s' and current directory.\n",
                FB_REFERENCE_DLL_DIR);
        fb_judge_shutdown();
        return 1;
    }

    fb_reference_init(ref_h);

    reference_vtable = fb_reference_backend();
    if (!reference_vtable) {
        fprintf(stderr, "[FATAL] fb_reference_backend() returned NULL\n");
        fb_judge_shutdown();
        return 1;
    }

    fb_judge_register_oracle(reference_vtable);
    fb_judge_register_backend(FB_BACKEND_ID_REFERENCE, reference_vtable);

    for (i = 0; i < sizeof(k_cases) / sizeof(k_cases[0]); i++) {
        if (run_case(&k_cases[i]) != 0) {
            fb_judge_shutdown();
            return 1;
        }
    }

    fb_judge_shutdown();
    printf("Result: PASS\n");
    return 0;
}