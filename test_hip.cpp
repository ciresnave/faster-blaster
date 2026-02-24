#include <hip/hip_runtime.h>
#include <stdio.h>

int main() {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    printf("hipGetDeviceCount returned: %d (0=success)\n", err);
    printf("Device count: %d\n", device_count);
    
    for (int i = 0; i < device_count; i++) {
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, i) == hipSuccess) {
            printf("Device %d: %s\n", i, prop.name);
            printf("  gcnArchName: %s\n", prop.gcnArchName);
        }
    }
    return 0;
}
