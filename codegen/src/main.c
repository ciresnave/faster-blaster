/**
 * faster-blaster Backend Wrapper Code Generator
 * 
 * Build-time tool to automatically extract function signatures from BLAS/LAPACK
 * backend headers and generate wrapper code.
 * 
 * Usage:
 *   ./fb_codegen \
 *       --headers-dir backends/headers \
 *       --templates-dir codegen/templates \
 *       --output-dir backends/generated \
 *       --config backends/codegen/backends_config.yaml
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
    #include <direct.h>
    #include <windows.h>
    #define mkdir(p, m) _mkdir(p)
#else
    #include <dirent.h>
    #include <sys/types.h>
    #include <sys/stat.h>
#endif

#include "extractor.h"
#include "generator.h"
#include "logging.h"

typedef struct {
    char headers_dir[512];
    char templates_dir[512];
    char output_dir[512];
    char config_file[512];
} ProgramConfig;

typedef struct {
    OperationList* ops;
    const char* backend;
} RecursiveWalkContext;

/* Recursively walk directory tree and extract from all .c and .h files */
#ifdef _WIN32
static void walk_directory_recursive_win32(const char* dir_path, RecursiveWalkContext* ctx) {
    WIN32_FIND_DATAA find_data;
    HANDLE find_handle;
    char search_path[1024];
    
    snprintf(search_path, sizeof(search_path), "%s\\*", dir_path);
    
    find_handle = FindFirstFileA(search_path, &find_data);
    if (find_handle == INVALID_HANDLE_VALUE) return;
    
    do {
        /* Skip . and .. */
        if (find_data.cFileName[0] == '.') continue;
        
        char full_path[1024];
        snprintf(full_path, sizeof(full_path), "%s\\%s", dir_path, find_data.cFileName);
        
        if (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            /* Recursively walk subdirectory */
            walk_directory_recursive_win32(full_path, ctx);
        } else {
            /* Check if file is .h or .c */
            const char* ext = strrchr(find_data.cFileName, '.');
            if (!ext || (strcmp(ext, ".h") != 0 && strcmp(ext, ".c") != 0)) {
                continue;
            }
            
            /* Try both conventions: Fortran (*_) first, then C (cblas_*) */
            OperationList* ops = extract_operations_from_header(full_path, ctx->backend, "fortran");
            if (!ops || ops->count == 0) {
                ops = extract_operations_from_header(full_path, ctx->backend, "c");
            }
            
            if (ops && ops->count > 0) {
                /* Merge into context ops */
                if (ctx->ops->count + ops->count > ctx->ops->capacity) {
                    ctx->ops->capacity *= 2;
                    ctx->ops->operations = realloc(ctx->ops->operations,
                                                   sizeof(Operation) * ctx->ops->capacity);
                }
                
                for (int i = 0; i < ops->count; i++) {
                    ctx->ops->operations[ctx->ops->count++] = ops->operations[i];
                }
                free(ops->operations);
                free(ops);
            }
        }
    } while (FindNextFileA(find_handle, &find_data));
    
    FindClose(find_handle);
}
#else
static void walk_directory_recursive_unix(const char* dir_path, RecursiveWalkContext* ctx) {
    DIR* dir = opendir(dir_path);
    if (!dir) return;
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        /* Skip . and .. */
        if (entry->d_name[0] == '.') continue;
        
        char full_path[1024];
        snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, entry->d_name);
        
        struct stat st;
        if (stat(full_path, &st) != 0) continue;
        
        if (S_ISDIR(st.st_mode)) {
            /* Recursively walk subdirectory */
            walk_directory_recursive_unix(full_path, ctx);
        } else if (S_ISREG(st.st_mode)) {
            /* Check if file is .h or .c */
            const char* ext = strrchr(entry->d_name, '.');
            if (!ext || (strcmp(ext, ".h") != 0 && strcmp(ext, ".c") != 0)) {
                continue;
            }
            
            /* Try both conventions: Fortran (*_) first, then C (cblas_*) */
            OperationList* ops = extract_operations_from_header(full_path, ctx->backend, "fortran");
            if (!ops || ops->count == 0) {
                ops = extract_operations_from_header(full_path, ctx->backend, "c");
            }
            
            if (ops && ops->count > 0) {
                /* Merge into context ops */
                if (ctx->ops->count + ops->count > ctx->ops->capacity) {
                    ctx->ops->capacity *= 2;
                    ctx->ops->operations = realloc(ctx->ops->operations,
                                                   sizeof(Operation) * ctx->ops->capacity);
                }
                
                for (int i = 0; i < ops->count; i++) {
                    ctx->ops->operations[ctx->ops->count++] = ops->operations[i];
                }
                free(ops->operations);
                free(ops);
            }
        }
    }
    closedir(dir);
}
#endif

static int parse_arguments(int argc, char** argv, ProgramConfig* config) {
    memset(config, 0, sizeof(ProgramConfig));
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--headers-dir") == 0 && i + 1 < argc) {
            strncpy(config->headers_dir, argv[++i], sizeof(config->headers_dir) - 1);
        } else if (strcmp(argv[i], "--templates-dir") == 0 && i + 1 < argc) {
            strncpy(config->templates_dir, argv[++i], sizeof(config->templates_dir) - 1);
        } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            strncpy(config->output_dir, argv[++i], sizeof(config->output_dir) - 1);
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            strncpy(config->config_file, argv[++i], sizeof(config->config_file) - 1);
        }
    }
    
    /* Validate */
    if (!config->headers_dir[0] || !config->output_dir[0]) {
        log_error("Missing required arguments");
        return 0;
    }
    
    return 1;
}

/* Find all backend header directories */
static int find_backends(const char* headers_dir, char** backend_names, int max_backends) {
#ifdef _WIN32
    WIN32_FIND_DATAA find_data;
    HANDLE find_handle;
    char search_path[1024];
    
    snprintf(search_path, sizeof(search_path), "%s\\*", headers_dir);
    
    find_handle = FindFirstFileA(search_path, &find_data);
    if (find_handle == INVALID_HANDLE_VALUE) {
        log_error("Cannot open headers directory: %s", headers_dir);
        return 0;
    }
    
    int count = 0;
    do {
        /* Skip . and .. */
        if (find_data.cFileName[0] == '.') continue;
        
        /* Check if it's a directory */
        if (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            backend_names[count] = malloc(256);
            strncpy(backend_names[count], find_data.cFileName, 255);
            log_info("✓ Found backend: %s", find_data.cFileName);
            count++;
            
            if (count >= max_backends) break;
        }
    } while (FindNextFileA(find_handle, &find_data));
    
    FindClose(find_handle);
    return count;
#else
    DIR* dir = opendir(headers_dir);
    if (!dir) {
        log_error("Cannot open headers directory: %s", headers_dir);
        return 0;
    }
    
    int count = 0;
    struct dirent* entry;
    
    while ((entry = readdir(dir)) != NULL && count < max_backends) {
        /* Skip . and .. */
        if (entry->d_name[0] == '.') continue;
        
        /* Check if it's a directory */
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", headers_dir, entry->d_name);
        
        struct stat st;
        if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
            backend_names[count] = malloc(256);
            strncpy(backend_names[count], entry->d_name, 255);
            log_info("✓ Found backend: %s", entry->d_name);
            count++;
        }
    }
    
    closedir(dir);
    return count;
#endif
}

int main(int argc, char** argv) {
    ProgramConfig config;
    
    log_info("faster-blaster Backend Wrapper Code Generator");
    log_info("===============================================");
    
    if (!parse_arguments(argc, argv, &config)) {
        log_error("Usage: fb_codegen --headers-dir <dir> --output-dir <dir> --templates-dir <dir>");
        return 1;
    }
    
    /* Create output directory */
    mkdir(config.output_dir, 0755);
    
    /* Find all backends */
    char* backend_names[32];
    int backend_count = find_backends(config.headers_dir, backend_names, 32);
    
    if (backend_count == 0) {
        log_warn("No backends found in %s", config.headers_dir);
        return 0;
    }
    
    log_info("Processing %d backends...", backend_count);
    
    /* Clear master include file */
    char master_path[1024];
    snprintf(master_path, sizeof(master_path), "%s/all_backends.h", config.output_dir);
    FILE* master = fopen(master_path, "w");
    if (master) {
        fprintf(master, "/**\n * GENERATED MASTER INCLUDE\n * DO NOT EDIT\n */\n\n");
        fprintf(master, "#ifndef FB_ALL_BACKENDS_H\n");
        fprintf(master, "#define FB_ALL_BACKENDS_H\n\n");
        fclose(master);
    }
    
    /* Process each backend */
    for (int b = 0; b < backend_count; b++) {
        const char* backend = backend_names[b];
        log_info("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        log_info("Processing backend: %s", backend);
        log_info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        /* Find all source files for this backend (recursively) */
        char backend_dir[1024];
        snprintf(backend_dir, sizeof(backend_dir), "%s\\%s", config.headers_dir, backend);
        
        /* Extract from each .h and .c file recursively */
        OperationList* all_ops = malloc(sizeof(OperationList));
        all_ops->operations = malloc(sizeof(Operation) * 4096);
        all_ops->count = 0;
        all_ops->capacity = 4096;
        
        RecursiveWalkContext ctx = {
            .ops = all_ops,
            .backend = backend
        };
        
#ifdef _WIN32
        walk_directory_recursive_win32(backend_dir, &ctx);
#else
        walk_directory_recursive_unix(backend_dir, &ctx);
#endif
        
        /* Generate wrapper files for this backend */
        if (all_ops->count > 0) {
            GenerationConfig gen_config = {
                .backend = backend,
                .operations = all_ops,
                .output_dir = config.output_dir,
                .templates_dir = config.templates_dir
            };
            
            if (generate_wrappers(&gen_config)) {
                /* Append to master include */
                char header_filename[256];
                snprintf(header_filename, sizeof(header_filename),
                         "operations_%s.h", backend);
                append_to_master_include(config.output_dir, "all_backends.h",
                                        header_filename);
                
                log_info("✅ %s: %d operations", backend, all_ops->count);
            }
        }
        
        if (all_ops->operations) free(all_ops->operations);
        free(all_ops);
        free((void*)backend_names[b]);
    }
    
    /* Close master include */
    master = fopen(master_path, "a");
    if (master) {
        fprintf(master, "\n#endif /* FB_ALL_BACKENDS_H */\n");
        fclose(master);
    }
    
    log_info("\n✅ Code generation complete!");
    return 0;
}
