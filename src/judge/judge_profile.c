/**
 * @file judge_profile.c
 * @brief Profile accumulator implementation; public query helper functions.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "judge_profile.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Private helpers
 * ========================================================================= */

/** qsort comparator: descending double order. */
static int cmp_double_desc(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da > db) return -1;
    if (da < db) return  1;
    return 0;
}

/** qsort comparator: ascending uint64_t order. */
static int cmp_u64_asc(const void *a, const void *b)
{
    uint64_t ua = *(const uint64_t *)a;
    uint64_t ub = *(const uint64_t *)b;
    if (ua < ub) return -1;
    if (ua > ub) return  1;
    return 0;
}

/** Clamp a uint64_t to uint32_t range. */
static uint32_t clamp_ns(uint64_t v)
{
    return (v > UINT32_MAX) ? UINT32_MAX : (uint32_t)v;
}

/* =========================================================================
 * fb_metric_accum_t
 * ========================================================================= */

void fb_metric_accum_init(fb_metric_accum_t *a)
{
    memset(a, 0, sizeof(*a));
}

void fb_metric_accum_add(fb_metric_accum_t *a, const fb_judge_case_result_t *r)
{
    if (a->count >= FB_PROFILE_ACCUM_MAX_CASES)
        return; /* silently ignore overflow — corpus size should never exceed cap */

    uint32_t i = a->count++;
    a->digits[i]          = r->digits;
    a->is_fatal[i]        = r->is_fatal;
    a->is_oracle_fatal[i] = r->is_oracle_fatal;

    if (r->is_fatal)
        a->fatal_count++;
    if (r->is_oracle_fatal)
        a->has_oracle_fatal = true;
}

void fb_metric_accum_finish(const fb_metric_accum_t *a, fb_metric_profile_t *out)
{
    memset(out, 0, sizeof(*out));
    out->has_fatal_failure = (a->fatal_count > 0);
    out->test_case_count   = a->count;

    if (a->count == 0)
        return;

    /* Collect non-fatal, non-oracle-fatal digit values into a sorted array. */
    double sorted[FB_PROFILE_ACCUM_MAX_CASES];
    uint32_t valid_count = 0;
    double   max_digits  = 0.0;
    double   min_digits  = 99.0;

    for (uint32_t i = 0; i < a->count; i++) {
        if (a->is_oracle_fatal[i])
            continue;  /* oracle bug; don't count toward score */
        double d = a->digits[i];
        sorted[valid_count++] = (a->is_fatal[i]) ? 0.0 : d;
        if (d > max_digits) max_digits = d;
        if (d < min_digits && !a->is_fatal[i]) min_digits = d;
    }

    if (valid_count == 0) {
        /* All cases were oracle-fatal — profile is meaningless. */
        out->has_fatal_failure = true;
        return;
    }

    /* Sort descending to build the accuracy curve. */
    qsort(sorted, valid_count, sizeof(double), cmp_double_desc);

    out->max_observed_digits = (uint8_t)fmin(max_digits, 16.0);

    /* guaranteed_digits: floor of the worst (last) non-fatal value. */
    {
        double worst = sorted[valid_count - 1];
        out->guaranteed_digits = (uint8_t)(worst < 0.0 ? 0 : (uint8_t)worst);
    }

    /* typical_digits: digits at the 50th percentile (median). */
    {
        uint32_t mid_idx = valid_count / 2;
        out->typical_digits = (uint8_t)fmin(sorted[mid_idx], 16.0);
    }

    /* Build the accuracy curve from the sorted data.
     *
     * For each sorted[i], pass_rate = (i+1) / valid_count.
     * We want points sorted descending in digits, ascending pass_rate.
     * We de-duplicate by only adding a point when the integer floor of 'digits'
     * changes, keeping the last occurrence (which has the highest pass_rate
     * for that digit level).
     *
     * The first entry is the highest-digit point (best case);
     * the last entry captures guaranteed_digits at pass_rate = 1.0.
     */
    uint32_t curve_len = 0;
    int8_t   last_digit_floor = -1;

    for (uint32_t i = 0; i < valid_count && curve_len < FB_JUDGE_MAX_CURVE_POINTS; i++) {
        int8_t dfloor = (int8_t)fmin(sorted[i], 16.0);
        float  prate  = (float)(i + 1) / (float)valid_count;

        if (dfloor != last_digit_floor) {
            /* New digit level — emit a new curve point. */
            out->curve[curve_len].digits    = (uint8_t)(dfloor < 0 ? 0 : dfloor);
            out->curve[curve_len].pass_rate = prate;
            curve_len++;
            last_digit_floor = dfloor;
        } else {
            /* Same digit floor — update pass_rate on the last emitted point. */
            out->curve[curve_len - 1].pass_rate = prate;
        }
    }
    out->curve_len = (uint8_t)curve_len;
}

/* =========================================================================
 * fb_timing_accum_t
 * ========================================================================= */

void fb_timing_accum_init(fb_timing_accum_t *a)
{
    memset(a, 0, sizeof(*a));
}

void fb_timing_accum_add(fb_timing_accum_t *a, uint64_t ns)
{
    if (a->count >= FB_PROFILE_ACCUM_MAX_CASES)
        return;
    a->ns[a->count++] = ns;
}

void fb_timing_accum_finish(const fb_timing_accum_t *a, fb_timing_profile_t *out)
{
    memset(out, 0, sizeof(*out));
    out->sample_count = a->count;

    if (a->count == 0)
        return;

    /* Sort ascending to compute percentiles. */
    uint64_t sorted[FB_PROFILE_ACCUM_MAX_CASES];
    memcpy(sorted, a->ns, a->count * sizeof(uint64_t));
    qsort(sorted, a->count, sizeof(uint64_t), cmp_u64_asc);

    out->min_ns = clamp_ns(sorted[0]);

    /* Mean */
    double sum = 0.0;
    for (uint32_t i = 0; i < a->count; i++)
        sum += (double)sorted[i];
    double mean = sum / (double)a->count;
    out->mean_ns = clamp_ns((uint64_t)mean);

    /* Standard deviation */
    double var_sum = 0.0;
    for (uint32_t i = 0; i < a->count; i++) {
        double diff = (double)sorted[i] - mean;
        var_sum += diff * diff;
    }
    out->stddev_ns = clamp_ns((uint64_t)sqrt(var_sum / (double)a->count));

    /* Percentiles (floor index). */
    uint32_t p50_idx = a->count / 2;
    uint32_t p95_idx = (a->count * 95) / 100;
    if (p95_idx >= a->count) p95_idx = a->count - 1;

    out->p50_ns = clamp_ns(sorted[p50_idx]);
    out->p95_ns = clamp_ns(sorted[p95_idx]);
}

/* =========================================================================
 * Combined profile builder (DIRECT archetype)
 * ========================================================================= */

void fb_profile_finish_direct(
    const fb_metric_accum_t  *direct_accum,
    const fb_timing_accum_t  *timing_accum,
    const fb_op_judge_meta_t *meta,
    fb_precision_profile_t   *out)
{
    /* Precision metrics — only 'direct' is populated for DIRECT archetype. */
    fb_metric_accum_finish(direct_accum, &out->direct);

    /* Timing */
    fb_timing_accum_finish(timing_accum, &out->timing);

    /* Summary */
    out->archetype       = (uint8_t)FB_JUDGE_DIRECT;
    out->limiting_metric = FB_JUDGE_LIMIT_DIRECT;

    /* Oracle ceiling from metadata (index 0 = F32, 1 = F64 as primary). */
    if (meta) {
        out->oracle_max_certifiable = meta->max_certifiable_f64;
    }
}

/* =========================================================================
 * Public: fb_judge_metric_digits_at_rate (declared in judge.h)
 * ========================================================================= */

uint8_t fb_judge_metric_digits_at_rate(
    const fb_metric_profile_t *metric,
    float                      required_pass_rate)
{
    if (!metric || metric->curve_len == 0)
        return 0;

    /*
     * The curve is sorted descending by digits (i.e. curve[0] = best case).
     * Walk until pass_rate >= required --- the last such entry is the answer.
     * If required_pass_rate == 0.0f, return the maximum observed digits.
     */
    uint8_t best = 0;
    for (uint32_t i = 0; i < metric->curve_len; i++) {
        if (metric->curve[i].pass_rate >= required_pass_rate) {
            /* Curve is descending: the first match might not be the weakest.
             * Continue to find the last curve point where the threshold is met.
             * Because digits decrease as we walk forward, the last match gives
             * the minimum digits still satisfying the pass_rate requirement. */
            best = metric->curve[i].digits;
        }
    }
    return best;
}

/* =========================================================================
 * Public: fb_judge_profile_meets_query (declared in judge.h)
 * ========================================================================= */

bool fb_judge_profile_meets_query(
    const fb_precision_profile_t *profile,
    const fb_judge_query_t       *query,
    uint8_t                      *effective_digits_out)
{
    if (!profile || !query) {
        if (effective_digits_out) *effective_digits_out = 0;
        return false;
    }

    uint8_t  min_digits = 255;  /* track worst across required metrics */
    bool     any_required = false;
    uint32_t mask = query->required_metrics;

    /* Evaluate each required metric. */
    struct { uint32_t bit; const fb_metric_profile_t *metric; } checks[] = {
        { FB_JUDGE_LIMIT_DIRECT,         &profile->direct         },
        { FB_JUDGE_LIMIT_VALUES,         &profile->values         },
        { FB_JUDGE_LIMIT_RECONSTRUCTION, &profile->reconstruction },
        { FB_JUDGE_LIMIT_ORTHOGONALITY,  &profile->orthogonality  },
        { FB_JUDGE_LIMIT_RESIDUAL,       &profile->residual       },
        { FB_JUDGE_LIMIT_PAIRS,          &profile->pairs          },
        { FB_JUDGE_LIMIT_SUBSPACE,       &profile->subspace       },
    };

    for (size_t ci = 0; ci < sizeof(checks)/sizeof(checks[0]); ci++) {
        if (!(mask & checks[ci].bit))
            continue;

        any_required = true;
        const fb_metric_profile_t *m = checks[ci].metric;

        if (m->test_case_count == 0) {
            /* Metric not populated — fail conservatively. */
            min_digits = 0;
            continue;
        }

        if (m->has_fatal_failure) {
            /* Any NaN/Inf/crash is a hard fail unless pass_rate allows it. */
            /* Count fatals against pass_rate implicitly via the curve. */
        }

        uint8_t d = fb_judge_metric_digits_at_rate(m, query->required_pass_rate);
        if (d < min_digits) min_digits = d;
    }

    if (!any_required) {
        /* No metrics required — trivially satisfied. */
        if (effective_digits_out) *effective_digits_out = 255;
        return true;
    }

    if (effective_digits_out)
        *effective_digits_out = (min_digits == 255) ? 0 : min_digits;

    return (min_digits != 255) && (min_digits >= query->requested_digits);
}
