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

static char* read_file_to_string(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buffer = malloc((size_t)size + 1);
    if (!buffer) {
        fclose(f);
        return NULL;
    }
    fread(buffer, 1, (size_t)size, f);
    buffer[size] = '\0';
    fclose(f);
    return buffer;
}

static char* replace_string(const char* source, const char* find, const char* replace) {
    size_t src_len = strlen(source);
    size_t find_len = strlen(find);
    size_t replace_len = strlen(replace);
    
    int count = 0;
    const char* tmp = source;
    while ((tmp = strstr(tmp, find)) != NULL) {
        count++;
        tmp += find_len;
    }
    
    if (count == 0) {
        return strdup(source);
    }
    
    size_t new_len = src_len + count * (replace_len - find_len);
    char* result = malloc(new_len + 1);
    if (!result) {
        return NULL;
    }
    
    char* dest = result;
    const char* curr = source;
    while ((tmp = strstr(curr, find)) != NULL) {
        size_t prefix_len = (size_t)(tmp - curr);
        memcpy(dest, curr, prefix_len);
        dest += prefix_len;
        memcpy(dest, replace, replace_len);
        dest += replace_len;
        curr = tmp + find_len;
    }
    strcpy(dest, curr);
    return result;
}

static char* render_operation_registry_template(
    const char* template_text,
    const char* backend,
    OperationList* ops
) {
    const char* categories[] = {
        "blas_level1",
        "blas_level2",
        "blas_level3",
        "lapack_auxiliary",
        "lapack_computational",
        "lapack_driver"
    };
    const char* category_names[] = {
        "BLAS Level 1",
        "BLAS Level 2",
        "BLAS Level 3",
        "LAPACK Auxiliary",
        "LAPACK Computational",
        "LAPACK Driver"
    };

    /* Replace simple placeholders */
    char* rendered = replace_string(template_text, "{{backend}}", backend);
    if (!rendered) return NULL;
    
    /* Replace operations count placeholder if present */
    char count_buf[32];
    snprintf(count_buf, sizeof(count_buf), "%d", ops->count);
    char* next = replace_string(rendered, "{{operations.length}}", count_buf);
    free(rendered);
    if (!next) return NULL;
    rendered = next;

    /* Find categories block */
    const char* cat_start = strstr(rendered, "{{#categories}}");
    const char* cat_end = strstr(rendered, "{{/categories}}");
    if (!cat_start || !cat_end || cat_end <= cat_start) {
        return rendered;
    }
    cat_start += strlen("{{#categories}}");
    size_t cat_block_len = (size_t)(cat_end - cat_start);
    char* cat_block = malloc(cat_block_len + 1);
    if (!cat_block) {
        free(rendered);
        return NULL;
    }
    strncpy(cat_block, cat_start, cat_block_len);
    cat_block[cat_block_len] = '\0';

    char* categories_output = malloc(1);
    categories_output[0] = '\0';
    size_t categories_output_len = 0;

    for (int cat_idx = 0; cat_idx < 6; cat_idx++) {
        const char* category = categories[cat_idx];
        int count = 0;
        for (int i = 0; i < ops->count; i++) {
            if (strcmp(ops->operations[i].category, category) == 0) count++;
        }
        if (count == 0) continue;

        /* Replace category placeholders */
        char* category_chunk = strdup(cat_block);
        if (!category_chunk) continue;
        char* temp = replace_string(category_chunk, "{{name}}", category_names[cat_idx]);
        free(category_chunk);
        if (!temp) continue;
        category_chunk = temp;
        snprintf(count_buf, sizeof(count_buf), "%d", count);
        temp = replace_string(category_chunk, "{{count}}", count_buf);
        free(category_chunk);
        if (!temp) continue;
        category_chunk = temp;

        /* Render operations block */
        const char* op_start = strstr(category_chunk, "{{#operations}}");
        const char* op_end = strstr(category_chunk, "{{/operations}}");
        if (!op_start || !op_end || op_end <= op_start) {
            free(category_chunk);
            continue;
        }
        op_start += strlen("{{#operations}}");
        size_t op_block_len = (size_t)(op_end - op_start);
        char* op_block = malloc(op_block_len + 1);
        if (!op_block) {
            free(category_chunk);
            continue;
        }
        strncpy(op_block, op_start, op_block_len);
        op_block[op_block_len] = '\0';

        size_t prefix_len = (size_t)(op_start - category_chunk - strlen("{{#operations}}"));
        size_t suffix_len = strlen(category_chunk) - (size_t)(op_end - category_chunk) - strlen("{{/operations}}");
        char* prefix = malloc(prefix_len + 1);
        char* suffix = malloc(suffix_len + 1);
        if (!prefix || !suffix) {
            free(op_block);
            free(category_chunk);
            free(prefix);
            free(suffix);
            continue;
        }
        strncpy(prefix, category_chunk, prefix_len);
        prefix[prefix_len] = '\0';
        strncpy(suffix, op_end + strlen("{{/operations}}"), suffix_len);
        suffix[suffix_len] = '\0';

        char* ops_rendered = malloc(1);
        ops_rendered[0] = '\0';
        size_t ops_rendered_len = 0;

        for (int i = 0; i < ops->count; i++) {
            if (strcmp(ops->operations[i].category, category) != 0) continue;
            char* op_chunk = strdup(op_block);
            if (!op_chunk) continue;
            temp = replace_string(op_chunk, "{{name}}", ops->operations[i].name);
            free(op_chunk);
            if (!temp) continue;
            op_chunk = temp;
            temp = replace_string(op_chunk, "{{return_type}}", ops->operations[i].return_type);
            free(op_chunk);
            if (!temp) continue;
            op_chunk = temp;
            temp = replace_string(op_chunk, "{{actual_name}}", ops->operations[i].actual_name);
            free(op_chunk);
            if (!temp) continue;
            op_chunk = temp;

            size_t op_chunk_len = strlen(op_chunk);
            ops_rendered = realloc(ops_rendered, ops_rendered_len + op_chunk_len + 1);
            memcpy(ops_rendered + ops_rendered_len, op_chunk, op_chunk_len);
            ops_rendered_len += op_chunk_len;
            ops_rendered[ops_rendered_len] = '\0';
            free(op_chunk);
        }

        free(op_block);

        /* Compose full category chunk */
        char* category_with_ops = malloc(prefix_len + ops_rendered_len + suffix_len + 1);
        memcpy(category_with_ops, prefix, prefix_len);
        memcpy(category_with_ops + prefix_len, ops_rendered, ops_rendered_len);
        memcpy(category_with_ops + prefix_len + ops_rendered_len, suffix, suffix_len);
        category_with_ops[prefix_len + ops_rendered_len + suffix_len] = '\0';

        size_t new_len = categories_output_len + strlen(category_with_ops);
        categories_output = realloc(categories_output, new_len + 1);
        memcpy(categories_output + categories_output_len, category_with_ops, strlen(category_with_ops));
        categories_output_len = new_len;
        categories_output[categories_output_len] = '\0';

        free(category_with_ops);
        free(prefix);
        free(suffix);
        free(ops_rendered);
        free(category_chunk);
    }

    /* Replace the categories block in the rendered template */
    char* after_categories = strstr(rendered, "{{#categories}}");
    if (!after_categories) {
        free(cat_block);
        free(rendered);
        free(categories_output);
        return NULL;
    }
    size_t head_len = (size_t)(after_categories - rendered);
    const char* tail_start = strstr(rendered, "{{/categories}}");
    if (!tail_start) {
        free(cat_block);
        free(rendered);
        free(categories_output);
        return NULL;
    }
    tail_start += strlen("{{/categories}}");
    size_t tail_len = strlen(tail_start);

    char* final_render = malloc(head_len + categories_output_len + tail_len + 1);
    memcpy(final_render, rendered, head_len);
    memcpy(final_render + head_len, categories_output, categories_output_len);
    memcpy(final_render + head_len + categories_output_len, tail_start, tail_len);
    final_render[head_len + categories_output_len + tail_len] = '\0';

    free(cat_block);
    free(rendered);
    free(categories_output);
    return final_render;
}

/* Generate operation registry file */
static int generate_operation_registry(
    const char* output_path,
    const char* backend,
    OperationList* ops,
    const char* templates_dir
) {
    if (templates_dir && templates_dir[0]) {
        char template_path[512];
        snprintf(template_path, sizeof(template_path), "%s/operation_registry.mustache", templates_dir);
        char* template_text = read_file_to_string(template_path);
        if (template_text) {
            char* rendered = render_operation_registry_template(template_text, backend, ops);
            free(template_text);
            if (rendered) {
                int success = write_file(output_path, rendered);
                free(rendered);
                if (success) {
                    log_info("✓ Generated: %s (%d operations)", output_path, ops->count);
                    return 1;
                }
            }
            log_warn("Template rendering failed; falling back to default generator");
        } else {
            log_warn("Template file not found: %s; falling back to default generator", template_path);
        }
    }
    
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
    
    if (!generate_operation_registry(output_path, config->backend, config->operations, config->templates_dir)) {
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
