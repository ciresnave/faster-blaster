/**
 * Code generation implementation
 */

#include "generator.h"
#include "logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
    #include <direct.h>
    #define mkdir(p, m) _mkdir(p)
#else
    #include <sys/types.h>
#endif

/* Write buffer to file */
static int write_file(const char* path, const char* content) {
    FILE* f = fopen(path, "w");
    if (!f) {
        log_error("Failed to open file for writing: %s", path);
        return 0;
    }
    
    fputs(content, f);
    fclose(f);
    
    log_debug("✓ Written: %s", path);
    return 1;
}

/* Generate operation registry file */
static int generate_operation_registry(
    const char* output_path,
    const char* backend,
    OperationList* ops
) {
    FILE* output = fopen(output_path, "w");
    if (!output) {
        log_error("Failed to open output file: %s", output_path);
        return 0;
    }
    
    /* Write header */
    fprintf(output, "/**\n");
    fprintf(output, " * GENERATED FILE - DO NOT EDIT\n");
    fprintf(output, " * Backend: %s\n", backend);
    fprintf(output, " * Operations: %d\n", ops->count);
    fprintf(output, " * Categories:\n");
    
    /* Count by category */
    int level1 = 0, level2 = 0, level3 = 0, lapack_aux = 0, lapack_comp = 0, lapack_driver = 0;
    for (int i = 0; i < ops->count; i++) {
        if (strcmp(ops->operations[i].category, "blas_level1") == 0) level1++;
        else if (strcmp(ops->operations[i].category, "blas_level2") == 0) level2++;
        else if (strcmp(ops->operations[i].category, "blas_level3") == 0) level3++;
        else if (strcmp(ops->operations[i].category, "lapack_auxiliary") == 0) lapack_aux++;
        else if (strcmp(ops->operations[i].category, "lapack_computational") == 0) lapack_comp++;
        else if (strcmp(ops->operations[i].category, "lapack_driver") == 0) lapack_driver++;
    }
    
    fprintf(output, " *   BLAS Level 1: %d\n", level1);
    fprintf(output, " *   BLAS Level 2: %d\n", level2);
    fprintf(output, " *   BLAS Level 3: %d\n", level3);
    fprintf(output, " *   LAPACK Auxiliary: %d\n", lapack_aux);
    fprintf(output, " *   LAPACK Computational: %d\n", lapack_comp);
    fprintf(output, " *   LAPACK Driver: %d\n", lapack_driver);
    fprintf(output, " */\n\n");
    
    fprintf(output, "#ifndef FB_OPERATIONS_%s_H\n", backend);
    fprintf(output, "#define FB_OPERATIONS_%s_H\n\n", backend);
    
    /* Write operations grouped by category */
    const char* categories[] = {
        "blas_level1",
        "blas_level2", 
        "blas_level3",
        "lapack_auxiliary",
        "lapack_computational",
        "lapack_driver"
    };
    
    for (int cat_idx = 0; cat_idx < 6; cat_idx++) {
        const char* category = categories[cat_idx];
        int count = 0;
        
        /* Count operations in this category */
        for (int i = 0; i < ops->count; i++) {
            if (strcmp(ops->operations[i].category, category) == 0) count++;
        }
        
        if (count == 0) continue;
        
        /* Write category header */
        fprintf(output, "/* %s: %d operations */\n", category, count);
        fprintf(output, "#ifdef OP\n");
        
        /* Write all operations in this category */
        for (int i = 0; i < ops->count; i++) {
            Operation* op = &ops->operations[i];
            if (strcmp(op->category, category) == 0) {
                fprintf(output, "OP(\n");
                fprintf(output, "    %s,  /* normalized name */\n", op->name);
                fprintf(output, "    %s,  /* return type */\n", op->return_type);
                fprintf(output, "    %s,  /* actual function */\n", op->actual_name);
                fprintf(output, "    (void)  /* parameters (simplified) */\n");
                fprintf(output, ")\n");
            }
        }
        
        fprintf(output, "#undef OP\n\n");
    }
    
    fprintf(output, "#endif /* FB_OPERATIONS_%s_H */\n", backend);
    fclose(output);
    
    log_info("✓ Generated: %s (%d operations)", output_path, ops->count);
    return 1;
}

int generate_wrappers(GenerationConfig* config) {
    char output_path[512];
    
    /* Generate operation registry */
    snprintf(output_path, sizeof(output_path), 
             "%s/operations_%s.h", config->output_dir, config->backend);
    
    if (!generate_operation_registry(output_path, config->backend, config->operations)) {
        return 0;
    }
    
    return 1;
}

int append_to_master_include(
    const char* output_dir,
    const char* master_file,
    const char* header_filename
) {
    char master_path[512];
    snprintf(master_path, sizeof(master_path), "%s/%s", output_dir, master_file);
    
    FILE* f = fopen(master_path, "a");
    if (!f) {
        log_error("Failed to open master include file: %s", master_path);
        return 0;
    }
    
    fprintf(f, "#include \"%s\"\n", header_filename);
    fclose(f);
    
    log_info("✓ Appended to master include: %s", header_filename);
    return 1;
}
