/**
 * @file exec_dag.c
 * @brief Execution DAG: layered DP planner for optimal backend path selection.
 *
 * Architecture:
 *   Each call to fb_exec_dag_plan():
 *   1. Snapshot: probe every registered plugin to get (score, device_type).
 *   2. Build: construct in-memory DP table [nsteps][nbackends].
 *   3. Solve:  forward DP -- for each (step, backend) cell compute the
 *              minimum cumulative cost accounting for execution cost at the
 *              node plus transfer cost on the incoming edge.
 *   4. Trace:  backtrack from the cheapest final-layer cell to recover the
 *              optimal backend sequence.
 *   5. Emit:   fill fb_exec_plan_t with per-step costs and transfer types.
 *
 * Cost model:
 *   exec_cost(backend, op_id) = FB_EXEC_BASE_NS / max(probe_score, 1)
 *     -- higher score = lower cost; baseline 100 ms / 100 = 1 us per op.
 *     -- TODO: replace with per-op judge profile when available.
 *
 *   xfer_cost(src_device, dst_device, size_bytes):
 *     H2D / D2H: size / 12 bytes-per-ns + 5000 ns latency overhead
 *     D2D:       size /  6 bytes-per-ns + 10000 ns latency overhead
 *     same:      0
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "../include/faster-blaster/backend_plugin.h"
#include "../include/faster-blaster/exec_dag.h"


/* =========================================================================
 * Constants
 * ========================================================================= */

/** Maximum number of plugins the DAG planner will consider. */
#define MAX_BACKENDS_DAG 16

/**
 * Baseline ns used to derive execution cost from a probe score.
 * exec_cost = FB_EXEC_BASE_NS / probe_score
 * score=100 => 1000 ns (1 us); score=1 => 100000 ns (100 us).
 */
#define FB_EXEC_BASE_NS 100000.0f

/** PCIe 4.0 x16 bidirectional rates, bytes per nanosecond. */
#define FB_H2D_BW 12.0f /* 12 GB/s */
#define FB_D2H_BW 12.0f
#define FB_D2D_BW 6.0f /* PCIe peer copy or slower NVLink path */

/** Fixed round-trip latency added per transfer (device command overhead). */
#define FB_H2D_LATENCY_NS 5000.0f
#define FB_D2H_LATENCY_NS 5000.0f
#define FB_D2D_LATENCY_NS 10000.0f

/* =========================================================================
 * Internal types
 * ========================================================================= */

/** Immutable snapshot of one backend's probe result. */
typedef struct {
  const fb_backend_plugin_t *plugin;
  int probe_score; /* 0-100 */
  fb_device_type_t device_type;
} backend_snapshot_t;

/** One DP cell: best cost to reach (step i, backend b) + backtrack link. */
typedef struct {
  float cost;        /* minimum cumulative cost to reach this cell     */
  int32_t prev_b;    /* backend index chosen at step i-1 (-1 at i==0) */
  float exec_ns;     /* execution cost at this node                    */
  float xfer_ns;     /* transfer cost on the incoming edge             */
  uint8_t xfer_type; /* FB_XFER_* for the incoming edge                */
} dp_cell_t;

/* =========================================================================
 * Helpers
 * ========================================================================= */

/**
 * Probe all registered plugins and build a snapshot array.
 * Returns the number of usable backends (score > 0).
 */
static uint32_t snapshot_backends(backend_snapshot_t *out, uint32_t max_out) {
  uint32_t n = 0;
  const fb_plugin_registry_entry_t *entry = fb_get_registered_plugins();

  while (entry && n < max_out) {
    const fb_backend_plugin_t *plugin = entry->plugin;

    if (plugin && plugin->probe && plugin->metadata) {
      fb_plugin_probe_result_t result = plugin->probe(NULL, NULL);

      if (result.score > 0) {
        out[n].plugin = plugin;
        out[n].probe_score = result.score;

        /* Map capabilities to device type */
        if (plugin->metadata->capabilities & FB_PLUGIN_CAP_GPU) {
          out[n].device_type = FB_DEVICE_GPU_0;
        } else {
          out[n].device_type = FB_DEVICE_CPU;
        }
        n++;
      }
    }
    entry = entry->next;
  }
  return n;
}

/**
 * Estimate execution cost for (backend, op_id) in nanoseconds.
 *
 * Current approximation: inverse probe score scaled to FB_EXEC_BASE_NS.
 * Future: consult judge store for per-op, per-backend, per-size profiles.
 */
static float exec_cost_ns(const backend_snapshot_t *b, uint32_t op_id,
                          fb_select_objective_t objective) {
  (void)op_id;
  (void)objective; /* same signal either way until judge profiles exist */

  int score = b->probe_score;
  if (score <= 0) {
    return FLT_MAX / 2.0f; /* effectively unavailable */
  }
  return FB_EXEC_BASE_NS / (float)score;
}

/**
 * Compute transfer cost and type for an edge (src_device -> dst_device)
 * carrying size_bytes of data.
 */
static void edge_cost(fb_device_type_t src, fb_device_type_t dst,
                      float size_bytes, float *cost_out,
                      fb_xfer_type_t *xtype_out) {
  if (src == dst) {
    *cost_out = 0.0f;
    *xtype_out = FB_XFER_NONE;
    return;
  }

  if (src == FB_DEVICE_CPU && dst != FB_DEVICE_CPU) {
    *cost_out = (size_bytes / FB_H2D_BW) + FB_H2D_LATENCY_NS;
    *xtype_out = FB_XFER_H2D;
  } else if (src != FB_DEVICE_CPU && dst == FB_DEVICE_CPU) {
    *cost_out = (size_bytes / FB_D2H_BW) + FB_D2H_LATENCY_NS;
    *xtype_out = FB_XFER_D2H;
  } else {
    /* GPU -> GPU */
    *cost_out = (size_bytes / FB_D2D_BW) + FB_D2D_LATENCY_NS;
    *xtype_out = FB_XFER_D2D;
  }
}

/**
 * Core DP solver shared by fb_exec_dag_plan() and fb_exec_dag_replan().
 *
 * @p steps / @p nsteps describe the sub-sequence to plan.
 * Returns a freshly-allocated plan covering exactly @p nsteps steps,
 * starting from step index 0 within the returned plan.
 */
static fb_exec_plan_t *run_dp(const fb_dag_step_input_t *steps, uint32_t nsteps,
                              fb_select_objective_t objective) {
  /* --- 1. Snapshot available backends ---------------------------------- */
  backend_snapshot_t backends[MAX_BACKENDS_DAG];
  uint32_t nb = snapshot_backends(backends, MAX_BACKENDS_DAG);

  /*
   * If no backends are registered (e.g. in unit tests without plugins),
   * synthesise a single-entry "trivial" backend so the planner still
   * returns a valid plan (all exec costs are 0, no transfers).
   */
  bool trivial_backend = false;
  if (nb == 0) {
    nb = 1;
    backends[0].plugin = NULL;
    backends[0].probe_score = 1;
    backends[0].device_type = FB_DEVICE_CPU;
    trivial_backend = true;
  }
  (void)trivial_backend;

  /* --- 2. Allocate DP table [nsteps][nb] ------------------------------- */
  dp_cell_t *dp = (dp_cell_t *)calloc((size_t)nsteps * nb, sizeof(dp_cell_t));
  if (!dp) {
    return NULL;
  }

  /* --- 3. Initialise layer 0 ------------------------------------------ */
  for (uint32_t b = 0; b < nb; b++) {
    float ec = exec_cost_ns(&backends[b], steps[0].op_id, objective);
    dp_cell_t *c = &dp[b];
    c->cost = ec;
    c->prev_b = -1;
    c->exec_ns = ec;
    c->xfer_ns = 0.0f;
    c->xfer_type = (uint8_t)FB_XFER_NONE;
  }

  /* --- 4. Fill layers 1 .. nsteps-1 ------------------------------------ */
  for (uint32_t i = 1; i < nsteps; i++) {
    float sz =
        (steps[i].size_bytes_hint > 0) ? (float)steps[i].size_bytes_hint : 0.0f;

    for (uint32_t b_next = 0; b_next < nb; b_next++) {
      float ec = exec_cost_ns(&backends[b_next], steps[i].op_id, objective);

      float best_cost = FLT_MAX;
      int32_t best_prev = 0;
      float best_xfer = 0.0f;
      fb_xfer_type_t best_xtype = FB_XFER_NONE;

      for (uint32_t b_prev = 0; b_prev < nb; b_prev++) {
        float xc;
        fb_xfer_type_t xt;
        edge_cost(backends[b_prev].device_type, backends[b_next].device_type,
                  sz, &xc, &xt);

        float prev_cost = dp[(i - 1) * nb + b_prev].cost;
        if (prev_cost >= FLT_MAX / 2.0f) {
          continue; /* previous cell was unreachable */
        }

        float candidate = prev_cost + xc + ec;
        if (candidate < best_cost) {
          best_cost = candidate;
          best_prev = (int32_t)b_prev;
          best_xfer = xc;
          best_xtype = xt;
        }
      }

      dp_cell_t *c = &dp[i * nb + b_next];
      c->cost = best_cost;
      c->prev_b = best_prev;
      c->exec_ns = ec;
      c->xfer_ns = best_xfer;
      c->xfer_type = (uint8_t)best_xtype;
    }
  }

  /* --- 5. Find best final backend ------------------------------------- */
  uint32_t best_final = 0;
  float best_final_cost = dp[(nsteps - 1) * nb + 0].cost;

  for (uint32_t b = 1; b < nb; b++) {
    if (dp[(nsteps - 1) * nb + b].cost < best_final_cost) {
      best_final_cost = dp[(nsteps - 1) * nb + b].cost;
      best_final = b;
    }
  }

  /* --- 6. Backtrack to recover the path -------------------------------- */
  uint32_t *path = (uint32_t *)malloc(nsteps * sizeof(uint32_t));
  if (!path) {
    free(dp);
    return NULL;
  }

  path[nsteps - 1] = best_final;
  for (int32_t i = (int32_t)nsteps - 2; i >= 0; i--) {
    path[i] = (uint32_t)dp[(i + 1) * nb + path[i + 1]].prev_b;
  }

  /* --- 7. Build the plan ---------------------------------------------- */
  fb_exec_plan_t *plan = (fb_exec_plan_t *)calloc(1, sizeof(*plan));
  if (!plan) {
    free(dp);
    free(path);
    return NULL;
  }

  plan->steps =
      (fb_dag_plan_step_t *)calloc(nsteps, sizeof(fb_dag_plan_step_t));
  if (!plan->steps) {
    free(dp);
    free(path);
    free(plan);
    return NULL;
  }

  plan->count = nsteps;
  plan->total_cost_ns = best_final_cost;
  plan->objective = objective;

  for (uint32_t i = 0; i < nsteps; i++) {
    dp_cell_t *c = &dp[i * nb + path[i]];
    plan->steps[i].backend_id = path[i];
    plan->steps[i].device_type = backends[path[i]].device_type;
    plan->steps[i].xfer_before = (fb_xfer_type_t)c->xfer_type;
    plan->steps[i].est_exec_ns = c->exec_ns;
    plan->steps[i].est_xfer_ns = c->xfer_ns;
  }

  free(dp);
  free(path);
  return plan;
}

/* =========================================================================
 * Public API
 * ========================================================================= */

fb_exec_plan_t *fb_exec_dag_plan(const fb_dag_step_input_t *steps,
                                 uint32_t nsteps,
                                 fb_select_objective_t objective) {
  if (!steps || nsteps == 0) {
    return NULL;
  }
  return run_dp(steps, nsteps, objective);
}

fb_exec_plan_t *fb_exec_dag_replan(const fb_exec_plan_t *current,
                                   uint32_t from_step,
                                   const fb_dag_step_input_t *steps,
                                   uint32_t nsteps,
                                   fb_select_objective_t objective) {
  if (!steps || nsteps == 0 || from_step >= nsteps) {
    return NULL;
  }

  uint32_t remaining = nsteps - from_step;

  /* Plan the remaining sub-sequence */
  fb_exec_plan_t *sub = run_dp(steps + from_step, remaining, objective);
  if (!sub) {
    return NULL;
  }

  /* Build merged plan: kept prefix from current + new suffix from sub */
  fb_exec_plan_t *merged = (fb_exec_plan_t *)calloc(1, sizeof(*merged));
  if (!merged) {
    fb_exec_plan_free(sub);
    return NULL;
  }

  merged->steps =
      (fb_dag_plan_step_t *)calloc(nsteps, sizeof(fb_dag_plan_step_t));
  if (!merged->steps) {
    free(merged);
    fb_exec_plan_free(sub);
    return NULL;
  }

  merged->count = nsteps;
  merged->objective = objective;

  /* Copy already-executed portion from the current plan */
  if (current && current->steps) {
    uint32_t copy_n = (from_step < current->count) ? from_step : current->count;
    for (uint32_t i = 0; i < copy_n; i++) {
      merged->steps[i] = current->steps[i];
    }
  }

  /* Copy the newly-planned suffix */
  for (uint32_t i = 0; i < sub->count; i++) {
    merged->steps[from_step + i] = sub->steps[i];
  }

  /* Recompute total cost */
  float total = 0.0f;
  for (uint32_t i = 0; i < nsteps; i++) {
    total += merged->steps[i].est_exec_ns + merged->steps[i].est_xfer_ns;
  }
  merged->total_cost_ns = total;

  fb_exec_plan_free(sub);
  return merged;
}

void fb_exec_plan_free(fb_exec_plan_t *plan) {
  if (!plan) {
    return;
  }
  free(plan->steps);
  free(plan);
}

const char *fb_xfer_type_name(fb_xfer_type_t xfer) {
  switch (xfer) {
  case FB_XFER_NONE:
    return "NONE";
  case FB_XFER_H2D:
    return "H2D";
  case FB_XFER_D2H:
    return "D2H";
  case FB_XFER_D2D:
    return "D2D";
  default:
    return "?";
  }
}

const char *fb_device_type_name(fb_device_type_t device) {
  switch (device) {
  case FB_DEVICE_CPU:
    return "CPU";
  case FB_DEVICE_GPU_0:
    return "GPU_0";
  case FB_DEVICE_GPU_1:
    return "GPU_1";
  case FB_DEVICE_GPU_2:
    return "GPU_2";
  case FB_DEVICE_GPU_3:
    return "GPU_3";
  default:
    return "?";
  }
}
