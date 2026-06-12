/**
 * @file power_manager.c
 * @brief Power Management Implementation
 */

#include "core/power_manager.h"
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <IOKit/ps/IOPowerSources.h>
#include <IOKit/ps/IOPSKeys.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

// Global power manager state
static fb_power_manager_t g_power_mgr = {0};
static bool g_power_mgr_initialized = false;

// ============================================================================
// Platform-Specific Power Source Detection
// ============================================================================

#ifdef _WIN32

static fb_power_source_t detect_power_source_windows(void) {
    SYSTEM_POWER_STATUS status;
    if (GetSystemPowerStatus(&status)) {
        if (status.ACLineStatus == 1) {
            return FB_POWER_AC;
        } else if (status.ACLineStatus == 0) {
            return (status.BatteryLifePercent > 20) ? 
                   FB_POWER_BATTERY : FB_POWER_BATTERY_LOW;
        }
    }
    return FB_POWER_UNKNOWN;
}

static uint32_t get_battery_percentage_windows(void) {
    SYSTEM_POWER_STATUS status;
    if (GetSystemPowerStatus(&status)) {
        return status.BatteryLifePercent;
    }
    return 0;
}

#elif defined(__APPLE__)

static fb_power_source_t detect_power_source_mac(void) {
    CFTypeRef ps_info = IOPSCopyPowerSourcesInfo();
    if (!ps_info) return FB_POWER_UNKNOWN;
    
    CFArrayRef ps_list = IOPSCopyPowerSourcesList(ps_info);
    if (!ps_list) {
        CFRelease(ps_info);
        return FB_POWER_UNKNOWN;
    }
    
    fb_power_source_t source = FB_POWER_AC; // Default to AC
    
    for (CFIndex i = 0; i < CFArrayGetCount(ps_list); i++) {
        CFTypeRef ps = CFArrayGetValueAtIndex(ps_list, i);
        CFDictionaryRef desc = IOPSGetPowerSourceDescription(ps_info, ps);
        
        if (desc) {
            CFStringRef state = CFDictionaryGetValue(desc, CFSTR(kIOPSPowerSourceStateKey));
            if (state && CFStringCompare(state, CFSTR(kIOPSBatteryPowerValue), 0) == kCFCompareEqualTo) {
                // On battery
                CFNumberRef capacity = CFDictionaryGetValue(desc, CFSTR(kIOPSCurrentCapacityKey));
                int percent = 0;
                if (capacity) {
                    CFNumberGetValue(capacity, kCFNumberIntType, &percent);
                }
                source = (percent > 20) ? FB_POWER_BATTERY : FB_POWER_BATTERY_LOW;
                break;
            }
        }
    }
    
    CFRelease(ps_list);
    CFRelease(ps_info);
    return source;
}

static uint32_t get_battery_percentage_mac(void) {
    CFTypeRef ps_info = IOPSCopyPowerSourcesInfo();
    if (!ps_info) return 0;
    
    CFArrayRef ps_list = IOPSCopyPowerSourcesList(ps_info);
    if (!ps_list) {
        CFRelease(ps_info);
        return 0;
    }
    
    uint32_t percent = 100;
    
    for (CFIndex i = 0; i < CFArrayGetCount(ps_list); i++) {
        CFTypeRef ps = CFArrayGetValueAtIndex(ps_list, i);
        CFDictionaryRef desc = IOPSGetPowerSourceDescription(ps_info, ps);
        
        if (desc) {
            CFNumberRef capacity = CFDictionaryGetValue(desc, CFSTR(kIOPSCurrentCapacityKey));
            if (capacity) {
                int val;
                CFNumberGetValue(capacity, kCFNumberIntType, &val);
                percent = val;
                break;
            }
        }
    }
    
    CFRelease(ps_list);
    CFRelease(ps_info);
    return percent;
}

#elif defined(__linux__)

static fb_power_source_t detect_power_source_linux(void) {
    // Try common battery status paths
    const char* status_paths[] = {
        "/sys/class/power_supply/BAT0/status",
        "/sys/class/power_supply/BAT1/status",
        "/sys/class/power_supply/battery/status",
        NULL
    };
    
    for (int i = 0; status_paths[i]; i++) {
        FILE* fp = fopen(status_paths[i], "r");
        if (fp) {
            char status[32];
            if (fgets(status, sizeof(status), fp)) {
                fclose(fp);
                
                if (strstr(status, "Charging") || strstr(status, "Full")) {
                    return FB_POWER_AC;
                } else if (strstr(status, "Discharging")) {
                    // Check battery percentage
                    const char* capacity_path = "/sys/class/power_supply/BAT0/capacity";
                    FILE* cap_fp = fopen(capacity_path, "r");
                    if (cap_fp) {
                        int percent;
                        if (fscanf(cap_fp, "%d", &percent) == 1) {
                            fclose(cap_fp);
                            return (percent > 20) ? FB_POWER_BATTERY : FB_POWER_BATTERY_LOW;
                        }
                        fclose(cap_fp);
                    }
                    return FB_POWER_BATTERY;
                }
            }
            fclose(fp);
        }
    }
    
    // No battery found - assume desktop with AC power
    return FB_POWER_AC;
}

static uint32_t get_battery_percentage_linux(void) {
    const char* capacity_paths[] = {
        "/sys/class/power_supply/BAT0/capacity",
        "/sys/class/power_supply/BAT1/capacity",
        "/sys/class/power_supply/battery/capacity",
        NULL
    };
    
    for (int i = 0; capacity_paths[i]; i++) {
        FILE* fp = fopen(capacity_paths[i], "r");
        if (fp) {
            int percent;
            if (fscanf(fp, "%d", &percent) == 1) {
                fclose(fp);
                return percent;
            }
            fclose(fp);
        }
    }
    
    return 100; // Default to full if unknown
}

#endif

// ============================================================================
// Initialization
// ============================================================================

int fb_power_manager_init(void) {
    if (g_power_mgr_initialized) {
        return 0;
    }
    
    memset(&g_power_mgr, 0, sizeof(g_power_mgr));
    
    // Detect initial power source
    fb_power_manager_update();
    
    g_power_mgr_initialized = true;
    return 0;
}

void fb_power_manager_shutdown(void) {
    if (!g_power_mgr_initialized) return;
    
    memset(&g_power_mgr, 0, sizeof(g_power_mgr));
    g_power_mgr_initialized = false;
}

// ============================================================================
// Power Source Detection
// ============================================================================

fb_power_source_t fb_get_power_source(void) {
    if (!g_power_mgr_initialized) {
        fb_power_manager_init();
    }
    
    return g_power_mgr.power_source;
}

uint32_t fb_get_battery_percentage(void) {
    if (!g_power_mgr_initialized) {
        fb_power_manager_init();
    }
    
    return g_power_mgr.battery_percentage;
}

bool fb_is_on_battery(void) {
    fb_power_source_t source = fb_get_power_source();
    return (source == FB_POWER_BATTERY || source == FB_POWER_BATTERY_LOW);
}

bool fb_is_battery_low(void) {
    return fb_get_power_source() == FB_POWER_BATTERY_LOW;
}

// ============================================================================
// State Update
// ============================================================================

int fb_power_manager_update(void) {
    if (!g_power_mgr_initialized) {
        fb_power_manager_init();
    }
    
#ifdef _WIN32
    g_power_mgr.power_source = detect_power_source_windows();
    g_power_mgr.battery_percentage = get_battery_percentage_windows();
#elif defined(__APPLE__)
    g_power_mgr.power_source = detect_power_source_mac();
    g_power_mgr.battery_percentage = get_battery_percentage_mac();
#elif defined(__linux__)
    g_power_mgr.power_source = detect_power_source_linux();
    g_power_mgr.battery_percentage = get_battery_percentage_linux();
#else
    g_power_mgr.power_source = FB_POWER_UNKNOWN;
    g_power_mgr.battery_percentage = 100;
#endif
    
    return 0;
}

// ============================================================================
// Power Preferences
// ============================================================================

void fb_set_power_mode(fb_power_mode_t mode) {
    if (!g_power_mgr_initialized) {
        fb_power_manager_init();
    }
    
    g_power_mgr.power_mode = mode;
}

fb_power_mode_t fb_get_power_mode(void) {
    if (!g_power_mgr_initialized) {
        fb_power_manager_init();
    }
    
    // Auto mode: choose based on power source
    if (g_power_mgr.power_mode == FB_POWER_MODE_AUTO) {
        if (fb_is_battery_low()) {
            return FB_POWER_MODE_SAVE;
        } else if (fb_is_on_battery()) {
            return FB_POWER_MODE_BALANCED;
        } else {
            return FB_POWER_MODE_PERFORMANCE;
        }
    }
    
    return g_power_mgr.power_mode;
}

bool fb_should_prefer_cpu(void) {
    // Prefer CPU when in power-saving mode or on battery
    fb_power_mode_t mode = fb_get_power_mode();
    
    if (mode == FB_POWER_MODE_SAVE) {
        return true;
    }
    
    if (mode == FB_POWER_MODE_BALANCED && fb_is_on_battery()) {
        return true;
    }
    
    return false;
}

// ============================================================================
// Energy Estimation
// ============================================================================

double fb_estimate_energy_cost(fb_compute_device_t* device, double execution_time_s) {
    if (!device) return 0.0;
    
    // Energy (Wh) = Power (W) * Time (h)
    double power_watts = device->power.tdp_watts;
    if (power_watts == 0) {
        // Estimate based on device type
        power_watts = (device->type == FB_DEVICE_GPU) ? 200.0 : 65.0;
    }
    
    double energy_wh = power_watts * (execution_time_s / 3600.0);
    return energy_wh;
}

fb_compute_device_t* fb_select_energy_efficient_device(
    fb_compute_device_t* devices, uint32_t num_devices,
    double estimated_runtime_s) {
    
    if (!devices || num_devices == 0) return NULL;
    
    fb_compute_device_t* best = NULL;
    double best_energy = 1e9;
    
    for (uint32_t i = 0; i < num_devices; i++) {
        fb_compute_device_t* dev = &devices[i];
        
        // Estimate execution time on this device
        // (This is simplified - real implementation would use performance models)
        double device_runtime = estimated_runtime_s;
        
        // Faster devices finish quicker
        if (dev->performance.fp32_gflops > 0) {
            double relative_speed = dev->performance.fp32_gflops / 1000.0;
            device_runtime = estimated_runtime_s / relative_speed;
        }
        
        double energy = fb_estimate_energy_cost(dev, device_runtime);
        
        if (energy < best_energy) {
            best_energy = energy;
            best = dev;
        }
    }
    
    return best;
}

// ============================================================================
// Power Budget Management
// ============================================================================

void fb_set_power_budget(uint32_t watts) {
    if (!g_power_mgr_initialized) {
        fb_power_manager_init();
    }
    
    g_power_mgr.power_budget_watts = watts;
}

uint32_t fb_get_power_budget(void) {
    if (!g_power_mgr_initialized) {
        fb_power_manager_init();
    }
    
    return g_power_mgr.power_budget_watts;
}

bool fb_device_within_power_budget(fb_compute_device_t* device) {
    if (!device) return false;
    
    uint32_t budget = fb_get_power_budget();
    if (budget == 0) return true; // No budget set
    
    uint32_t device_power = device->power.tdp_watts;
    if (device_power == 0) device_power = 100; // Default estimate
    
    return device_power <= budget;
}

// ============================================================================
// Debug/Print
// ============================================================================

void fb_print_power_status(void) {
    if (!g_power_mgr_initialized) {
        fb_power_manager_init();
    }
    
    printf("=== Power Status ===\n");
    
    const char* source_str;
    switch (g_power_mgr.power_source) {
        case FB_POWER_AC: source_str = "AC Power"; break;
        case FB_POWER_BATTERY: source_str = "Battery"; break;
        case FB_POWER_BATTERY_LOW: source_str = "Battery (Low)"; break;
        case FB_POWER_UPS: source_str = "UPS"; break;
        default: source_str = "Unknown"; break;
    }
    
    printf("Power Source: %s\n", source_str);
    
    if (fb_is_on_battery()) {
        printf("Battery: %u%%\n", g_power_mgr.battery_percentage);
    }
    
    const char* mode_str;
    fb_power_mode_t mode = fb_get_power_mode();
    switch (mode) {
        case FB_POWER_MODE_PERFORMANCE: mode_str = "Performance"; break;
        case FB_POWER_MODE_BALANCED: mode_str = "Balanced"; break;
        case FB_POWER_MODE_SAVE: mode_str = "Power Save"; break;
        default: mode_str = "Auto"; break;
    }
    
    printf("Power Mode: %s\n", mode_str);
    printf("Prefer CPU: %s\n", fb_should_prefer_cpu() ? "Yes" : "No");
    
    if (g_power_mgr.power_budget_watts > 0) {
        printf("Power Budget: %u W\n", g_power_mgr.power_budget_watts);
    }
}

const char* fb_power_source_string(fb_power_source_t source) {
    switch (source) {
        case FB_POWER_AC: return "AC";
        case FB_POWER_BATTERY: return "Battery";
        case FB_POWER_BATTERY_LOW: return "Battery (Low)";
        case FB_POWER_UPS: return "UPS";
        default: return "Unknown";
    }
}
