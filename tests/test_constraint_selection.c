/**
 * @file test_constraint_selection.c
 * @brief Unit tests for fb_select_backend_with_constraints()
 *
 * Tests the constraint-based backend selection layer added in dispatch_tables.c.
 * Uses an empty ranked-dispatch table (no profiles on disk) so all entry counts
 * are 0, meaning fb_ranked_get_best() returns the table's fallback_backend_id.
 *
 * This lets us exercise every control-flow branch of
 * fb_select_backend_with_constraints() without requiring actual benchmark
 * profile files on disk.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "../include/dispatch_tables.h"
#include "../include/faster-blaster/ranked_dispatch.h"
#include "../include/faster-blaster/backend_ids.h"
#include "../src/judge/judge_op_ids.h"

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Minimal test harness
 * ------------------------------------------------------------------------- */

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT_EQ(a, b, label)                                              \
    do {                                                                    \
        if ((uint32_t)(a) == (uint32_t)(b)) {                               \
            printf("  PASS  %s\n", (label));                                \
            g_pass++;                                                       \
        } else {                                                            \
            printf("  FAIL  %s  (got %u, expected %u)\n",                  \
                   (label), (uint32_t)(a), (uint32_t)(b));                  \
            g_fail++;                                                       \
        }                                                                   \
    } while (0)

#define EXPECT_NE(a, b, label)                                              \
    do {                                                                    \
        if ((uint32_t)(a) != (uint32_t)(b)) {                               \
            printf("  PASS  %s\n", (label));                                \
            g_pass++;                                                       \
        } else {                                                            \
            printf("  FAIL  %s  (got %u, did not expect it)\n",             \
                   (label), (uint32_t)(a));                                 \
            g_fail++;                                                       \
        }                                                                   \
    } while (0)

/* An arbitrary backend ID we can tell apart from FB_BACKEND_ID_REFERENCE. */
#define FAKE_FALLBACK 42u

/* A shared empty ranked table used by most tests.
 * Built once: call fb_ranked_table_build() with a non-existent profile dir
 * so every (op, criterion) slot ends up with 0 entries.  Queries therefore
 * always return the table's fallback_backend_id. */
static fb_ranked_table_t *g_empty_table = NULL;

/* Build the shared empty table. */
static void setup_empty_table(void)
{
    /* Non-existent directory — every fb_judge_load_profile call will fail,
     * resulting in 0 ranked entries for every operation. */
    const char *empty_dir = "C:\\fb_test_no_such_profile_dir_xyzzy_9999";

    /* One fake backend in the pool; fallback = FAKE_FALLBACK. */
    uint32_t bids[1] = { FAKE_FALLBACK };
    g_empty_table = fb_ranked_table_build(
        empty_dir, 0u, 0u, bids, 1u, FAKE_FALLBACK);
}

static void teardown_empty_table(void)
{
    fb_ranked_table_free(g_empty_table);
    g_empty_table = NULL;
}

/* Restore the global dispatch table after each test group. */
static fb_ranked_table_t *g_saved_table = NULL;

static void save_global_table(void)
{
    g_saved_table = (fb_ranked_table_t *)fb_op_dispatch_get_table();
    fb_op_dispatch_set_table(NULL);
}

static void restore_global_table(void)
{
    fb_op_dispatch_set_table(g_saved_table);
    g_saved_table = NULL;
}

/* -------------------------------------------------------------------------
 * Test 1: No global table registered → FB_BACKEND_ID_REFERENCE
 * ------------------------------------------------------------------------- */
static void test_no_table(void)
{
    printf("Test 1: No global dispatch table registered\n");
    save_global_table();
    /* Table is NULL after save_global_table(). */

    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SAXPY, NULL),
              FB_BACKEND_ID_REFERENCE,
              "NULL table, NULL constraints → reference");

    fb_dispatch_constraints_t c = {0};
    c.prefer_criterion = FB_OPTIMIZE_FASTEST;
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SAXPY, &c),
              FB_BACKEND_ID_REFERENCE,
              "NULL table, fast criterion → reference");

    restore_global_table();
}

/* -------------------------------------------------------------------------
 * Test 2: NULL constraints with empty table → fallback (balanced)
 * ------------------------------------------------------------------------- */
static void test_null_constraints(void)
{
    printf("Test 2: NULL constraints \u2192 balanced fallback\n");
    save_global_table();
    fb_op_dispatch_set_table(g_empty_table);

    /* No entries in table → fb_ranked_get_best returns FAKE_FALLBACK. */
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SAXPY, NULL),
              FAKE_FALLBACK,
              "NULL constraints → balanced → fallback");

    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SGEMM, NULL),
              FAKE_FALLBACK,
              "NULL constraints, different op → fallback");

    restore_global_table();
}

/* -------------------------------------------------------------------------
 * Test 3: No hard constraints — prefer_criterion routing
 * ------------------------------------------------------------------------- */
static void test_prefer_criterion_routing(void)
{
    printf("Test 3: prefer_criterion routing (no hard constraints)\n");
    save_global_table();
    fb_op_dispatch_set_table(g_empty_table);

    fb_dispatch_constraints_t c = {0};

    c.prefer_criterion = FB_OPTIMIZE_FASTEST;
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SAXPY, &c),
              FAKE_FALLBACK, "prefer FASTEST → fallback");

    c.prefer_criterion = FB_OPTIMIZE_POWER_EFFICIENT;
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SAXPY, &c),
              FAKE_FALLBACK, "prefer POWER_EFFICIENT → fallback");

    c.prefer_criterion = FB_OPTIMIZE_ACCURACY;
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SAXPY, &c),
              FAKE_FALLBACK, "prefer ACCURACY → fallback");

    c.prefer_criterion = FB_OPTIMIZE_PRECISION;
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SAXPY, &c),
              FAKE_FALLBACK, "prefer PRECISION → fallback");

    c.prefer_criterion = FB_OPTIMIZE_BALANCED;
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SAXPY, &c),
              FAKE_FALLBACK, "prefer BALANCED → fallback");

    restore_global_table();
}

/* -------------------------------------------------------------------------
 * Test 4: Hard latency constraint only — no entries pass → degraded fallback
 * ------------------------------------------------------------------------- */
static void test_latency_constraint_only(void)
{
    printf("Test 4: Hard latency constraint only (empty table \u2192 degrade)\n");
    save_global_table();
    fb_op_dispatch_set_table(g_empty_table);

    fb_dispatch_constraints_t c = {0};
    c.max_time_us = 1000u;          /* 1 ms ceiling */
    c.require_exact_precision = false;
    c.prefer_criterion = FB_OPTIMIZE_BALANCED;

    /* No ranked entries → loop is empty → degraded → fallback */
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SAXPY, &c),
              FAKE_FALLBACK,
              "latency constraint, empty table, degrade → fallback");

    restore_global_table();
}

/* -------------------------------------------------------------------------
 * Test 5: Hard accuracy constraint only — no entries pass → degraded fallback
 * ------------------------------------------------------------------------- */
static void test_accuracy_constraint_only(void)
{
    printf("Test 5: Hard accuracy constraint only (empty table \u2192 degrade)\n");
    save_global_table();
    fb_op_dispatch_set_table(g_empty_table);

    fb_dispatch_constraints_t c = {0};
    c.min_accuracy = 200u;          /* ~12 significant digits */
    c.require_exact_precision = false;
    c.prefer_criterion = FB_OPTIMIZE_FASTEST;

    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SAXPY, &c),
              FAKE_FALLBACK,
              "accuracy constraint, empty table, degrade → fallback");

    restore_global_table();
}

/* -------------------------------------------------------------------------
 * Test 6: min_precision > min_accuracy — stricter one is used
 * ------------------------------------------------------------------------- */
static void test_precision_wins_over_accuracy(void)
{
    printf("Test 6: min_precision stricter than min_accuracy\n");
    save_global_table();
    fb_op_dispatch_set_table(g_empty_table);

    fb_dispatch_constraints_t c = {0};
    c.min_accuracy  = 50u;   /* ~3 digits */
    c.min_precision = 250u;  /* ~15 digits — stricter */
    c.require_exact_precision = false;
    c.prefer_criterion = FB_OPTIMIZE_BALANCED;

    /* Still degrades to fallback because table is empty. */
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SAXPY, &c),
              FAKE_FALLBACK,
              "precision stricter: empty table → degrade → fallback");

    restore_global_table();
}

/* -------------------------------------------------------------------------
 * Test 7: require_exact_precision=true, no entries → reference (no degrade)
 * ------------------------------------------------------------------------- */
static void test_require_exact_no_degrade(void)
{
    printf("Test 7: require_exact_precision=true \u2192 no degraded fallback\n");
    save_global_table();
    fb_op_dispatch_set_table(g_empty_table);

    fb_dispatch_constraints_t c = {0};
    c.min_accuracy = 128u;          /* ~7.5 digits */
    c.require_exact_precision = true;

    /* No entries → falls through → reference (hard failure) */
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SAXPY, &c),
              FB_BACKEND_ID_REFERENCE,
              "require_exact=true, empty table → reference");

    /* Same with latency constraint */
    c.min_accuracy = 0u;
    c.max_time_us  = 500u;
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SAXPY, &c),
              FB_BACKEND_ID_REFERENCE,
              "latency + require_exact=true, empty table → reference");

    /* Same with both hard constraints */
    c.min_accuracy = 128u;
    c.max_time_us  = 500u;
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SAXPY, &c),
              FB_BACKEND_ID_REFERENCE,
              "both constraints + require_exact=true, empty table → reference");

    restore_global_table();
}

/* -------------------------------------------------------------------------
 * Test 8: zero constraints struct (all fields 0) → balanced fallback
 * ------------------------------------------------------------------------- */
static void test_zero_constraints_struct(void)
{
    printf("Test 8: Zero-initialised constraints \u2192 balanced fallback\n");
    save_global_table();
    fb_op_dispatch_set_table(g_empty_table);

    fb_dispatch_constraints_t c;
    memset(&c, 0, sizeof(c));

    /* All fields zero: no hard constraints, prefer_criterion=0 (FASTEST). */
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SAXPY, &c),
              FAKE_FALLBACK,
              "zero-init constraints → fallback");

    restore_global_table();
}

/* -------------------------------------------------------------------------
 * Test 9: after unregistering table → reference backend again
 * ------------------------------------------------------------------------- */
static void test_unregister_table(void)
{
    printf("Test 9: Unregister table \u2192 reference backend\n");
    save_global_table();

    fb_op_dispatch_set_table(g_empty_table);
    fb_dispatch_constraints_t c = {0};
    c.prefer_criterion = FB_OPTIMIZE_FASTEST;
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SAXPY, &c),
              FAKE_FALLBACK, "table registered → fallback");

    fb_op_dispatch_set_table(NULL);
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SAXPY, &c),
              FB_BACKEND_ID_REFERENCE, "table unregistered → reference");

    restore_global_table();
}

/* -------------------------------------------------------------------------
 * Test 10: degrade with different prefer_criterion values
 * ------------------------------------------------------------------------- */
static void test_degrade_respects_prefer_criterion(void)
{
    printf("Test 10: Degraded mode respects prefer_criterion\n");
    save_global_table();
    fb_op_dispatch_set_table(g_empty_table);

    /* With an empty table every fb_ranked_get_best() returns FAKE_FALLBACK
     * regardless of criterion. Verify all branches run without crash. */
    fb_dispatch_constraints_t c = {0};
    c.min_accuracy            = 255u;  /* hard constraint that can't be met */
    c.require_exact_precision = false;

    c.prefer_criterion = FB_OPTIMIZE_FASTEST;
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SGEMM, &c),
              FAKE_FALLBACK, "degrade FASTEST → fallback");

    c.prefer_criterion = FB_OPTIMIZE_ACCURACY;
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SGEMM, &c),
              FAKE_FALLBACK, "degrade ACCURACY → fallback");

    c.prefer_criterion = FB_OPTIMIZE_PRECISION;
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SGEMM, &c),
              FAKE_FALLBACK, "degrade PRECISION → fallback");

    c.prefer_criterion = FB_OPTIMIZE_POWER_EFFICIENT;
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SGEMM, &c),
              FAKE_FALLBACK, "degrade POWER → fallback");

    c.prefer_criterion = FB_OPTIMIZE_BALANCED;
    EXPECT_EQ(fb_select_backend_with_constraints(FB_OP_SGEMM, &c),
              FAKE_FALLBACK, "degrade BALANCED → fallback");

    restore_global_table();
}

/* -------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */
int main(void)
{
    printf("=== test_constraint_selection ===\n\n");

    setup_empty_table();

    if (!g_empty_table) {
        printf("FATAL: fb_ranked_table_build() returned NULL — cannot continue\n");
        return 1;
    }

    test_no_table();
    test_null_constraints();
    test_prefer_criterion_routing();
    test_latency_constraint_only();
    test_accuracy_constraint_only();
    test_precision_wins_over_accuracy();
    test_require_exact_no_degrade();
    test_zero_constraints_struct();
    test_unregister_table();
    test_degrade_respects_prefer_criterion();

    teardown_empty_table();

    printf("\n------------------------------------------\n");
    printf("Total:  %d\n", g_pass + g_fail);
    printf("Passed: %d\n", g_pass);
    printf("Failed: %d\n", g_fail);
    if (g_fail == 0)
        printf("Success Rate: 100%%  -- ALL TESTS PASSED\n");
    else
        printf("Success Rate: %.1f%%\n",
               100.0 * g_pass / (g_pass + g_fail));

    return (g_fail == 0) ? 0 : 1;
}
