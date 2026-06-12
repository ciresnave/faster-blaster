#include <stdio.h>

extern "C" {
    struct fb_gpu_backend_trait;
    const fb_gpu_backend_trait* fb_onemkl_get_trait(void);
}

int main() {
    printf("Calling fb_onemkl_get_trait...\n");
    const fb_gpu_backend_trait* trait = fb_onemkl_get_trait();
    printf("Got trait: %p\n", (void*)trait);
    return 0;
}
