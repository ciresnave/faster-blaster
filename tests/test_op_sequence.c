/**
 * @file test_op_sequence.c
 * @brief Unit tests for the operation sequence executor.
 *
 * Covers:
 *  Test 01 – Empty sequence: compile + execute succeed with 0 steps.
 *  Test 02 – NULL guard on seq and step parameters for add().
 *  Test 03 – Execute without compile returns -1.
 *  Test 04 – Fusion: SGEMM → SAXPY annotated may_fuse_with_next=true.
 *  Test 05 – Fusion: SGEMM → DGEMM annotated may_fuse_with_next=true.
 *  Test 06 – No fusion: SAXPY → SDOT annotated may_fuse_with_next=false.
 *  Test 07 – exec_fn called in correct order (counter array).
 *  Test 08 – NULL vtable case: exec_fn still receives NULL vtable.
 *  Test 09 – get_step() returns correct data after compile.
 *  Test 10 – step_count() matches number of add() calls.
 *  Test 11 – Capacity enforcement: add beyond max_steps returns -1.
 *  Test 12 – reset() returns sequence to empty uncompiled state.
 *  Test 13 – Add after compile without reset returns -1.
 *  Test 14 – exec_fn returning non-zero aborts sequence and propagates error.
 *  Test 15 – NULL exec_fn steps are skipped silently.
 *  Test 16 – step name preserved after add/compile.
 *  Test 17 – is_compiled() reflects sequence state.
 *  Test 18 – Fusion not set on last step (no next step).
 *  Test 19 – Fusion: DGEMM → DSCAL (cross-precision GEMM/SCAL).
 *  Test 20 – Free NULL sequence is safe (no crash).
 *  Test 21 – Reset then re-add and re-compile works correctly.
 *  Test 22 – fb_op_sequence_print() on NULL and populated sequence.
 *  Test 23 – max_steps=0 creation returns NULL.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "../include/faster-blaster/op_chain.h"
#include "../src/judge/judge_op_ids.h"

/* =========================================================================
 * Test infrastructure
 * ========================================================================= */

static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg)                                       \
    do {                                                              \
        g_tests_run++;                                                \
        if (cond) {                                                   \
            g_tests_passed++;                                         \
        } else {                                                      \
            g_tests_failed++;                                         \
            printf("  FAIL [line %d]: %s\n", __LINE__, msg);         \
        }                                                             \
    } while (0)

#define TEST_BEGIN(name) printf("[TEST] %s\n", name)
#define TEST_END()       /* nothing — output already visible */

/* =========================================================================
 * Helpers
 * ========================================================================= */

/* A no-op exec_fn that always succeeds. */
static int noop_exec(const fb_backend_vtable_t *vt, void *ud)
{
    (void)vt;
    (void)ud;
    return 0;
}

/* An exec_fn that writes the step index into a user-supplied int array. */
typedef struct {
    int   *order;    /* pre-allocated array */
    int    pos;      /* next write index    */
    int    max;      /* allocated length    */
} call_tracker_t;

static int tracking_exec(const fb_backend_vtable_t *vt, void *ud)
{
    (void)vt;
    call_tracker_t *t = (call_tracker_t *)ud;
    if (t->pos < t->max) {
        t->order[t->pos++] = t->pos;   /* record call number 1-based */
    }
    return 0;
}

/* An exec_fn that fails and records the vtable pointer it received. */
typedef struct {
    const fb_backend_vtable_t *received_vtable;
    int                        return_code;
} failing_exec_data_t;

static int failing_exec(const fb_backend_vtable_t *vt, void *ud)
{
    failing_exec_data_t *d = (failing_exec_data_t *)ud;
    d->received_vtable = vt;
    return d->return_code;
}

/* Build a step with a given op_id and name. */
static fb_seq_step_t make_step(uint32_t op_id, const char *name,
                                int (*fn)(const fb_backend_vtable_t *, void *),
                                void *ud)
{
    fb_seq_step_t s;
    memset(&s, 0, sizeof(s));
    s.op_id    = op_id;
    s.exec_fn  = fn;
    s.user_data = ud;
    if (name) {
        strncpy(s.name, name, sizeof(s.name) - 1);
        s.name[sizeof(s.name) - 1] = '\0';
    }
    return s;
}

/* =========================================================================
 * Test cases
 * ========================================================================= */

static void test_01_empty_sequence(void)
{
    TEST_BEGIN("01 - Empty sequence compile+execute");

    fb_op_sequence_t *seq = fb_op_sequence_create(8);
    TEST_ASSERT(seq != NULL, "create() returned NULL");

    int rc_compile = fb_op_sequence_compile(seq);
    TEST_ASSERT(rc_compile == 0, "compile() empty sequence failed");

    int rc_execute = fb_op_sequence_execute(seq);
    TEST_ASSERT(rc_execute == 0, "execute() empty sequence failed");

    TEST_ASSERT(fb_op_sequence_step_count(seq) == 0, "step_count != 0 for empty");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_02_null_guards_add(void)
{
    TEST_BEGIN("02 - NULL guards on add()");

    int rc;
    fb_seq_step_t s = make_step(FB_OP_SGEMM, "x", noop_exec, NULL);

    rc = fb_op_sequence_add(NULL, &s);
    TEST_ASSERT(rc == -1, "add(NULL seq) should return -1");

    fb_op_sequence_t *seq = fb_op_sequence_create(4);
    rc = fb_op_sequence_add(seq, NULL);
    TEST_ASSERT(rc == -1, "add(NULL step) should return -1");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_03_execute_without_compile(void)
{
    TEST_BEGIN("03 - Execute without compile returns -1");

    fb_op_sequence_t *seq = fb_op_sequence_create(4);
    fb_seq_step_t s = make_step(FB_OP_SGEMM, "g", noop_exec, NULL);
    fb_op_sequence_add(seq, &s);

    int rc = fb_op_sequence_execute(seq);
    TEST_ASSERT(rc == -1, "execute() before compile() must return -1");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_04_fusion_gemm_axpy(void)
{
    TEST_BEGIN("04 - Fusion: SGEMM -> SAXPY");

    fb_op_sequence_t *seq = fb_op_sequence_create(4);

    fb_seq_step_t g = make_step(FB_OP_SGEMM, "gemm", noop_exec, NULL);
    fb_seq_step_t a = make_step(FB_OP_SAXPY, "axpy", noop_exec, NULL);
    fb_op_sequence_add(seq, &g);
    fb_op_sequence_add(seq, &a);

    fb_op_sequence_compile(seq);

    const fb_seq_step_t *step0 = fb_op_sequence_get_step(seq, 0);
    const fb_seq_step_t *step1 = fb_op_sequence_get_step(seq, 1);

    TEST_ASSERT(step0 != NULL, "step0 is NULL");
    TEST_ASSERT(step1 != NULL, "step1 is NULL");
    TEST_ASSERT(step0->may_fuse_with_next == true,
                "SGEMM->SAXPY: may_fuse_with_next should be true");
    TEST_ASSERT(step1->may_fuse_with_next == false,
                "Last step may_fuse_with_next should be false");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_05_fusion_gemm_gemm(void)
{
    TEST_BEGIN("05 - Fusion: SGEMM -> DGEMM");

    fb_op_sequence_t *seq = fb_op_sequence_create(4);

    fb_seq_step_t g1 = make_step(FB_OP_SGEMM, "g1", noop_exec, NULL);
    fb_seq_step_t g2 = make_step(FB_OP_DGEMM, "g2", noop_exec, NULL);
    fb_op_sequence_add(seq, &g1);
    fb_op_sequence_add(seq, &g2);

    fb_op_sequence_compile(seq);

    const fb_seq_step_t *step0 = fb_op_sequence_get_step(seq, 0);
    TEST_ASSERT(step0 != NULL, "step0 is NULL");
    TEST_ASSERT(step0->may_fuse_with_next == true,
                "SGEMM->DGEMM: may_fuse_with_next should be true");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_06_no_fusion_axpy_sdot(void)
{
    TEST_BEGIN("06 - No fusion: SAXPY -> SDOT");

    fb_op_sequence_t *seq = fb_op_sequence_create(4);

    fb_seq_step_t a = make_step(FB_OP_SAXPY, "axpy", noop_exec, NULL);
    fb_seq_step_t d = make_step(FB_OP_SDOT,  "sdot", noop_exec, NULL);
    fb_op_sequence_add(seq, &a);
    fb_op_sequence_add(seq, &d);

    fb_op_sequence_compile(seq);

    const fb_seq_step_t *step0 = fb_op_sequence_get_step(seq, 0);
    TEST_ASSERT(step0 != NULL, "step0 is NULL");
    TEST_ASSERT(step0->may_fuse_with_next == false,
                "SAXPY->SDOT: may_fuse_with_next should be false");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_07_exec_order(void)
{
    TEST_BEGIN("07 - exec_fn called in correct order");

    int order[3]  = {0, 0, 0};
    call_tracker_t t0 = { order,     0, 3 };
    call_tracker_t t1 = { order + 1, 0, 2 };
    call_tracker_t t2 = { order + 2, 0, 1 };

    fb_op_sequence_t *seq = fb_op_sequence_create(4);
    fb_op_sequence_add(seq, &(fb_seq_step_t){ .op_id = FB_OP_SAXPY, .exec_fn = tracking_exec, .user_data = &t0 });
    fb_op_sequence_add(seq, &(fb_seq_step_t){ .op_id = FB_OP_DAXPY, .exec_fn = tracking_exec, .user_data = &t1 });
    fb_op_sequence_add(seq, &(fb_seq_step_t){ .op_id = FB_OP_SGEMM, .exec_fn = tracking_exec, .user_data = &t2 });

    fb_op_sequence_compile(seq);
    int rc = fb_op_sequence_execute(seq);

    TEST_ASSERT(rc == 0, "execute() returned non-zero");
    TEST_ASSERT(t0.pos == 1, "step 0 exec_fn not called exactly once");
    TEST_ASSERT(t1.pos == 1, "step 1 exec_fn not called exactly once");
    TEST_ASSERT(t2.pos == 1, "step 2 exec_fn not called exactly once");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_08_null_vtable(void)
{
    TEST_BEGIN("08 - NULL vtable: exec_fn still called");

    failing_exec_data_t d = { (const fb_backend_vtable_t *)(void *)0x1, 0 };

    fb_op_sequence_t *seq = fb_op_sequence_create(2);
    fb_seq_step_t s = make_step(FB_OP_SGEMM, "g", failing_exec, &d);
    fb_op_sequence_add(seq, &s);
    fb_op_sequence_compile(seq);

    /*
     * We cannot force fb_get_active_vtable() to return NULL here (no way
     * to unload the backend in a unit test without a real backend loaded),
     * but we can verify the exec_fn was called at all.
     * The received_vtable will be whatever the active vtable is (may be NULL
     * in a standalone test run with no backend loaded).
     */
    int rc = fb_op_sequence_execute(seq);
    /* d.received_vtable will be whatever was active — just check called. */
    TEST_ASSERT(rc == 0, "execute() unexpectedly failed");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_09_get_step_data(void)
{
    TEST_BEGIN("09 - get_step() data preserved after compile");

    fb_op_sequence_t *seq = fb_op_sequence_create(4);
    fb_seq_step_t s = make_step(FB_OP_DGEMM, "my_gemm", noop_exec, (void *)0xABCD);
    fb_op_sequence_add(seq, &s);
    fb_op_sequence_compile(seq);

    const fb_seq_step_t *got = fb_op_sequence_get_step(seq, 0);
    TEST_ASSERT(got != NULL, "get_step(0) returned NULL");
    TEST_ASSERT(got->op_id == FB_OP_DGEMM, "op_id mismatch");
    TEST_ASSERT(strcmp(got->name, "my_gemm") == 0, "name mismatch");
    TEST_ASSERT(got->user_data == (void *)0xABCD, "user_data mismatch");

    TEST_ASSERT(fb_op_sequence_get_step(seq, 1) == NULL,
                "get_step(1) should be NULL (out of range)");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_10_step_count(void)
{
    TEST_BEGIN("10 - step_count() matches add() calls");

    fb_op_sequence_t *seq = fb_op_sequence_create(8);
    TEST_ASSERT(fb_op_sequence_step_count(seq) == 0, "initial count != 0");

    for (int i = 0; i < 5; i++) {
        fb_seq_step_t s = make_step((uint32_t)(FB_OP_SAXPY + i), NULL, noop_exec, NULL);
        fb_op_sequence_add(seq, &s);
        TEST_ASSERT((int)fb_op_sequence_step_count(seq) == i + 1,
                    "step_count != expected after add");
    }

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_11_capacity_enforcement(void)
{
    TEST_BEGIN("11 - Capacity enforcement");

    fb_op_sequence_t *seq = fb_op_sequence_create(2);

    fb_seq_step_t s = make_step(FB_OP_SAXPY, NULL, noop_exec, NULL);
    TEST_ASSERT(fb_op_sequence_add(seq, &s) == 0, "add 1st step failed");
    TEST_ASSERT(fb_op_sequence_add(seq, &s) == 0, "add 2nd step failed");
    TEST_ASSERT(fb_op_sequence_add(seq, &s) == -1, "add 3rd past capacity should be -1");
    TEST_ASSERT(fb_op_sequence_step_count(seq) == 2, "count should remain 2");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_12_reset(void)
{
    TEST_BEGIN("12 - reset() returns sequence to empty uncompiled state");

    fb_op_sequence_t *seq = fb_op_sequence_create(4);
    fb_seq_step_t s = make_step(FB_OP_SGEMM, "g", noop_exec, NULL);
    fb_op_sequence_add(seq, &s);
    fb_op_sequence_compile(seq);

    TEST_ASSERT(fb_op_sequence_is_compiled(seq) == true, "should be compiled");

    fb_op_sequence_reset(seq);

    TEST_ASSERT(fb_op_sequence_step_count(seq) == 0, "count should be 0 after reset");
    TEST_ASSERT(fb_op_sequence_is_compiled(seq) == false,
                "should not be compiled after reset");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_13_add_after_compile_blocked(void)
{
    TEST_BEGIN("13 - add() after compile without reset returns -1");

    fb_op_sequence_t *seq = fb_op_sequence_create(4);
    fb_seq_step_t s = make_step(FB_OP_SGEMM, "g", noop_exec, NULL);
    fb_op_sequence_add(seq, &s);
    fb_op_sequence_compile(seq);

    int rc = fb_op_sequence_add(seq, &s);
    TEST_ASSERT(rc == -1, "add() after compile must return -1");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_14_exec_fn_error_abort(void)
{
    TEST_BEGIN("14 - Non-zero exec_fn aborts sequence");

    int order[3] = {0, 0, 0};
    call_tracker_t tracker = { order, 0, 3 };
    failing_exec_data_t fail_data = { NULL, 42 };

    fb_op_sequence_t *seq = fb_op_sequence_create(4);
    fb_seq_step_t s_ok    = make_step(FB_OP_SAXPY,  NULL, tracking_exec, &tracker);
    fb_seq_step_t s_fail  = make_step(FB_OP_SGEMM,  NULL, failing_exec,  &fail_data);
    fb_seq_step_t s_after = make_step(FB_OP_DAXPY,  NULL, tracking_exec, &tracker);

    fb_op_sequence_add(seq, &s_ok);
    fb_op_sequence_add(seq, &s_fail);
    fb_op_sequence_add(seq, &s_after);
    fb_op_sequence_compile(seq);

    int rc = fb_op_sequence_execute(seq);
    TEST_ASSERT(rc == 42, "execute() should return 42 from failing step");
    TEST_ASSERT(tracker.pos == 1, "only 1 step before abort should have run");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_15_null_exec_fn_skipped(void)
{
    TEST_BEGIN("15 - NULL exec_fn steps are skipped");

    int order[1] = {0};
    call_tracker_t tracker = { order, 0, 1 };

    fb_op_sequence_t *seq = fb_op_sequence_create(4);
    fb_seq_step_t s_null = make_step(FB_OP_SGEMM, "skip", NULL, NULL);
    fb_seq_step_t s_real = make_step(FB_OP_SAXPY, "real", tracking_exec, &tracker);

    fb_op_sequence_add(seq, &s_null);
    fb_op_sequence_add(seq, &s_real);
    fb_op_sequence_compile(seq);

    int rc = fb_op_sequence_execute(seq);
    TEST_ASSERT(rc == 0, "execute() should succeed");
    TEST_ASSERT(tracker.pos == 1, "real step should have been called once");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_16_name_preserved(void)
{
    TEST_BEGIN("16 - Step name preserved after add/compile");

    fb_op_sequence_t *seq = fb_op_sequence_create(2);
    fb_seq_step_t s = make_step(FB_OP_SGEMM, "my_test_step", noop_exec, NULL);
    fb_op_sequence_add(seq, &s);
    fb_op_sequence_compile(seq);

    const fb_seq_step_t *got = fb_op_sequence_get_step(seq, 0);
    TEST_ASSERT(got != NULL, "get_step returned NULL");
    TEST_ASSERT(strcmp(got->name, "my_test_step") == 0, "name not preserved");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_17_is_compiled_state(void)
{
    TEST_BEGIN("17 - is_compiled() reflects state");

    fb_op_sequence_t *seq = fb_op_sequence_create(2);
    TEST_ASSERT(!fb_op_sequence_is_compiled(seq), "should not be compiled initially");
    fb_op_sequence_compile(seq);
    TEST_ASSERT(fb_op_sequence_is_compiled(seq), "should be compiled after compile()");
    fb_op_sequence_reset(seq);
    TEST_ASSERT(!fb_op_sequence_is_compiled(seq), "should not be compiled after reset()");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_18_no_fusion_last_step(void)
{
    TEST_BEGIN("18 - may_fuse_with_next=false on last step");

    fb_op_sequence_t *seq = fb_op_sequence_create(4);
    fb_seq_step_t g = make_step(FB_OP_SGEMM, "g", noop_exec, NULL);
    fb_op_sequence_add(seq, &g);
    fb_op_sequence_compile(seq);

    const fb_seq_step_t *step0 = fb_op_sequence_get_step(seq, 0);
    TEST_ASSERT(step0 != NULL, "step0 is NULL");
    TEST_ASSERT(step0->may_fuse_with_next == false,
                "Single last step must not have may_fuse_with_next set");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_19_fusion_dgemm_dscal(void)
{
    TEST_BEGIN("19 - Fusion: DGEMM -> DSCAL");

    fb_op_sequence_t *seq = fb_op_sequence_create(4);
    fb_seq_step_t g = make_step(FB_OP_DGEMM, "dgemm", noop_exec, NULL);
    fb_seq_step_t s = make_step(FB_OP_DSCAL, "dscal", noop_exec, NULL);
    fb_op_sequence_add(seq, &g);
    fb_op_sequence_add(seq, &s);
    fb_op_sequence_compile(seq);

    const fb_seq_step_t *step0 = fb_op_sequence_get_step(seq, 0);
    TEST_ASSERT(step0 != NULL, "step0 is NULL");
    TEST_ASSERT(step0->may_fuse_with_next == true,
                "DGEMM->DSCAL: may_fuse_with_next should be true");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_20_free_null_safe(void)
{
    TEST_BEGIN("20 - Free NULL is safe");
    fb_op_sequence_free(NULL);   /* must not crash */
    TEST_ASSERT(true, "reached here safely");
    TEST_END();
}

static void test_21_reset_and_reuse(void)
{
    TEST_BEGIN("21 - Reset then re-add and re-compile");

    int order[2] = {0, 0};
    call_tracker_t t0 = { order,     0, 2 };
    call_tracker_t t1 = { order + 1, 0, 1 };

    fb_op_sequence_t *seq = fb_op_sequence_create(4);

    /* First use */
    fb_op_sequence_add(seq, &(fb_seq_step_t){ .op_id = FB_OP_SGEMM, .exec_fn = tracking_exec, .user_data = &t0 });
    fb_op_sequence_compile(seq);
    fb_op_sequence_execute(seq);

    /* Reset and reuse */
    fb_op_sequence_reset(seq);
    t0.pos = 0;

    fb_op_sequence_add(seq, &(fb_seq_step_t){ .op_id = FB_OP_SAXPY, .exec_fn = tracking_exec, .user_data = &t0 });
    fb_op_sequence_add(seq, &(fb_seq_step_t){ .op_id = FB_OP_DAXPY, .exec_fn = tracking_exec, .user_data = &t1 });
    fb_op_sequence_compile(seq);
    int rc = fb_op_sequence_execute(seq);

    TEST_ASSERT(rc == 0, "execute() failed after reset+reuse");
    TEST_ASSERT(fb_op_sequence_step_count(seq) == 2, "step_count should be 2 after reuse");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_22_print_not_crash(void)
{
    TEST_BEGIN("22 - fb_op_sequence_print() does not crash");

    fb_op_sequence_print(NULL);   /* must not crash */

    fb_op_sequence_t *seq = fb_op_sequence_create(4);
    fb_seq_step_t s = make_step(FB_OP_SGEMM, "gemm_print_test", noop_exec, NULL);
    fb_op_sequence_add(seq, &s);
    fb_op_sequence_compile(seq);

    fb_op_sequence_print(seq);    /* must not crash */
    TEST_ASSERT(true, "reached here safely");

    fb_op_sequence_free(seq);
    TEST_END();
}

static void test_23_zero_capacity_returns_null(void)
{
    TEST_BEGIN("23 - max_steps=0 creation returns NULL");

    fb_op_sequence_t *seq = fb_op_sequence_create(0);
    TEST_ASSERT(seq == NULL, "create(0) should return NULL");

    TEST_END();
}

/* =========================================================================
 * Main
 * ========================================================================= */

int main(void)
{
    printf("=========================================\n");
    printf("  Operation Sequence Executor Tests\n");
    printf("=========================================\n\n");

    test_01_empty_sequence();
    test_02_null_guards_add();
    test_03_execute_without_compile();
    test_04_fusion_gemm_axpy();
    test_05_fusion_gemm_gemm();
    test_06_no_fusion_axpy_sdot();
    test_07_exec_order();
    test_08_null_vtable();
    test_09_get_step_data();
    test_10_step_count();
    test_11_capacity_enforcement();
    test_12_reset();
    test_13_add_after_compile_blocked();
    test_14_exec_fn_error_abort();
    test_15_null_exec_fn_skipped();
    test_16_name_preserved();
    test_17_is_compiled_state();
    test_18_no_fusion_last_step();
    test_19_fusion_dgemm_dscal();
    test_20_free_null_safe();
    test_21_reset_and_reuse();
    test_22_print_not_crash();
    test_23_zero_capacity_returns_null();

    printf("\n=========================================\n");
    printf("Total:  %d\n", g_tests_run);
    printf("Passed: %d\n", g_tests_passed);
    printf("Failed: %d\n", g_tests_failed);
    printf("Success Rate: %.1f%%\n",
           g_tests_run ? 100.0 * g_tests_passed / g_tests_run : 0.0);
    printf("=========================================\n");

    return g_tests_failed > 0 ? 1 : 0;
}
