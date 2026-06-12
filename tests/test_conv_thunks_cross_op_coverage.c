#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

static void dummy_slot(void)
{
    /* Intentionally empty: used as a generic function-pointer payload. */
}

int main(void)
{
    fb_backend_vtable_t vtable;
    uint32_t op = 0;
    int f2c_installed = 0;
    int c2f_installed = 0;
    int no_op_verified = 0;
    int no_override_verified = 0;

    for (op = 0; op < FB_JUDGE_MAX_OPERATIONS; ++op) {
        fb_generic_fn fortran_seed = (fb_generic_fn)(void (*)(void))dummy_slot;
        fb_generic_fn cblas_seed = (fb_generic_fn)(void (*)(void))main;

        memset(&vtable, 0, sizeof(vtable));
        vtable.ext_ops[op][FB_CONV_FORTRAN] = fortran_seed;
        fb_install_conv_thunks(&vtable, op);
        if (vtable.ext_ops[op][FB_CONV_CBLAS] != NULL) {
            f2c_installed += 1;
        }

        memset(&vtable, 0, sizeof(vtable));

        vtable.ext_ops[op][FB_CONV_CBLAS] = cblas_seed;
        fb_install_conv_thunks(&vtable, op);
        if (vtable.ext_ops[op][FB_CONV_FORTRAN] != NULL) {
            c2f_installed += 1;
        }

        memset(&vtable, 0, sizeof(vtable));
        fb_install_conv_thunks(&vtable, op);
        if (vtable.ext_ops[op][FB_CONV_CBLAS] != NULL ||
            vtable.ext_ops[op][FB_CONV_FORTRAN] != NULL) {
            fprintf(stderr, "[FAIL] Unexpected install with no source slot for op %u\n", op);
            return 1;
        }
        no_op_verified += 1;

        memset(&vtable, 0, sizeof(vtable));
        vtable.ext_ops[op][FB_CONV_CBLAS] = cblas_seed;
        vtable.ext_ops[op][FB_CONV_FORTRAN] = fortran_seed;
        fb_install_conv_thunks(&vtable, op);
        if (vtable.ext_ops[op][FB_CONV_CBLAS] != cblas_seed ||
            vtable.ext_ops[op][FB_CONV_FORTRAN] != fortran_seed) {
            fprintf(stderr, "[FAIL] Existing slots were overridden for op %u\n", op);
            return 1;
        }
        no_override_verified += 1;
    }

    if (f2c_installed == 0 || c2f_installed == 0) {
        fprintf(stderr,
                "[FAIL] No cross-op installs observed (f2c=%d, c2f=%d)\n",
                f2c_installed, c2f_installed);
        return 1;
    }

    printf("[PASS] Observed Fortran->CBLAS installs for %d operations\n", f2c_installed);
    printf("[PASS] Observed CBLAS->Fortran installs for %d operations\n", c2f_installed);
    printf("[PASS] Verified no-op behavior for %d operations\n", no_op_verified);
    printf("[PASS] Verified no override behavior for %d operations\n", no_override_verified);
    printf("Result: PASS\n");
    return 0;
}
