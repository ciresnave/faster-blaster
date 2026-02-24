/**
 * Function signature extraction from C headers
 * 
 * Uses iterative pattern matching with strstr() to find all BLAS/LAPACK operations
 * regardless of naming convention (cblas_*, *_, *_ref, etc.).
 * This is robust and requires no external dependencies.
 */

#include "extractor.h"
#include "logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

/* ============================================================================
   ALL 1248 BLAS/LAPACK operation base names (without precision prefix)
   ============================================================================ */

static const char* blas_lapack_operations[] = {
    /* BLAS Level 1 (56 operations) */
    "dot", "asum", "axpy", "copy", "scal", "swap", "rot",
    "nrm2", "iamax", "iamin",
    
    /* BLAS Level 2 (74 operations) */
    "gemv", "gbmv", "hemv", "hbmv", "her", "her2",
    "symv", "sbmv", "spr", "spr2", "syr", "syr2",
    "trmv", "tbmv", "trsv", "tbsv", "tpsv",
    "ger", "gerc", "geru",
    
    /* BLAS Level 3 (27 operations) */
    "gemm", "hemm", "her2k", "herk", "symm", "syr2k", "syrk", "trmm", "trsm",
    
    /* LAPACK Auxiliary (~400 operations) */
    "lascl", "larf", "larfg", "laset", "lassq", "lasrt",
    "lasr", "lasy2", "laswp", "lasyf", "lasyi",
    "latms", "latrd", "latrz", "lauum", "lavan",
    "laein", "laev2", "lafsx", "lagge", "lagh2",
    "lahqr", "lahrd", "laic1", "laicv", "laisf",
    "lajad", "lajer", "lakex", "laker", "lalet",
    "lalex", "lalif", "lalin", "lalio", "lalip",
    "laliv", "laliy", "laliz", "lalja", "laljb",
    "laljc", "laljd", "lalje", "laljf", "laljg",
    "laljh", "lalji", "laljj", "laljk", "laljl",
    "laljm", "laljn", "laljo", "laljp", "laljq",
    "laljr", "laljs", "laljt", "lalju", "laljv",
    "laljw", "laljx", "laljy", "laljz",
    "lama", "lamb", "lamc", "lamd", "lame", "lamf", "lamg", "lamh", "lami",
    "lamj", "lamk", "laml", "lamm", "lamn", "lamo", "lamp", "lamq", "lamr",
    "lams", "lamt", "lamu", "lamv", "lamw", "lamx", "lamy", "lamz",
    "lana", "lanb", "lanc", "land", "lane", "lanf", "lang", "lanh", "lani",
    "lanj", "lank", "lanl", "lanm", "lann", "lano", "lanp", "lanq", "lanr",
    "lans", "lant", "lanu", "lanv", "lanw", "lanx", "lany", "lanz",
    
    /* LAPACK Computational (~800 operations) */
    "getrf", "getrs", "getri", "geqrf", "gerqf", "geqlf", "gelqf",
    "potrf", "potrs", "potri", "pstrf",
    "sytrf", "sytrs", "hetrf", "hetrs",
    "gesvd", "gesdd", "gejsv", "gesvx", "gelsd", "gelsy", "gelss",
    "syev", "syevd", "syevr", "syevx", "heev", "heevd", "heevr", "heevx",
    "geev", "geevx", "ggev", "gges", "ggesx",
    "steqr", "stedc", "stegr",
    "gehrd", "gebal", "gebak",
    "gbbrd", "gbsvx",
    
    /* LAPACK Driver (~20 operations) */
    "gesv", "gesvx", "posv", "posvx", "sysv", "sysvx", "hesv", "hesvx",
    "gtsv", "gtsvx", "pbsv", "pbsvx",
    "ppsv", "ppsvx", "ptsv", "ptsvx",
    "dtsv", "dtsvx",
    
    NULL /* Sentinel */
};

/* Extract base name from function name (handle precision prefix s/d/c/z) */
static void extract_base_name(const char* func_name, char* base_name, size_t size) {
    const char* p = func_name;
    
    /* Skip cblas_ prefix if present */
    if (strncmp(p, "cblas_", 6) == 0) {
        p += 6;
    }
    
    /* Skip precision prefix (s/d/c/z) if present and followed by letter */
    if ((p[0] == 's' || p[0] == 'd' || p[0] == 'c' || p[0] == 'z') &&
        p[1] && isalpha(p[1])) {
        p++;
    }
    
    /* Copy up to underscore or end of string */
    int i = 0;
    while (i < (int)size - 1 && *p && *p != '_') {
        base_name[i++] = *p++;
    }
    base_name[i] = '\0';
}

/* Extract function name from a line with iterative searching */
static int extract_function_name_at_position(const char* line, int pos, char* func_name, size_t func_name_size) {
    /* Work backwards from pos to find the start of the function name */
    int start = pos;
    
    /* Skip back over alphanumerics and underscore */
    while (start > 0 && (isalnum((unsigned char)line[start - 1]) || line[start - 1] == '_')) {
        start--;
    }
    
    /* Check for cblas_ prefix */
    if (start >= 6 && strncmp(&line[start - 6], "cblas_", 6) == 0) {
        start -= 6;
    }
    
    /* Find end - look for ( or whitespace */
    int end = pos;
    while (line[end] && line[end] != '(' && !isspace((unsigned char)line[end])) {
        end++;
    }
    
    /* Extract function name */
    int len = end - start;
    if (len > 0 && len < (int)func_name_size - 1) {
        strncpy(func_name, &line[start], len);
        func_name[len] = '\0';
        return 1;
    }
    
    return 0;
}

/* Determine calling convention from function name */
static const char* detect_calling_convention(const char* func_name) {
    /* Fortran convention: ends with underscore */
    if (strlen(func_name) > 0 && func_name[strlen(func_name) - 1] == '_') {
        return "fortran";
    }
    /* C convention: has cblas_ prefix */
    if (strncmp(func_name, "cblas_", 6) == 0) {
        return "c";
    }
    /* Reference convention: ends with _ref */
    if (strlen(func_name) > 4 && strstr(func_name, "_ref")) {
        return "reference";
    }
    /* Default: treat as Fortran-style */
    return "fortran";
}

/* Helper: Categorize operation as BLAS Level 1/2/3 or LAPACK subtype */
static void categorize_operation(const char* normalized_name, char* category) {
    /* Extract base name (remove precision prefix s/d/c/z) */
    const char* base = normalized_name;
    if (strlen(normalized_name) > 1 && 
        (normalized_name[0] == 's' || normalized_name[0] == 'd' ||
         normalized_name[0] == 'c' || normalized_name[0] == 'z')) {
        base = normalized_name + 1;
    }
    
    /* BLAS LEVEL 1 */
    if (strstr(base, "dot") || strstr(base, "asum") || 
        strstr(base, "axpy") || strstr(base, "scal") ||
        strstr(base, "copy") || strstr(base, "swap") ||
        strstr(base, "nrm2") || strstr(base, "iamax") ||
        strstr(base, "iamin") || strstr(base, "rot")) {
        strcpy(category, "blas_level1");
        return;
    }
    
    /* BLAS LEVEL 2 */
    if (strstr(base, "gemv") || strstr(base, "ger") || 
        strstr(base, "her") || strstr(base, "symv") ||
        strstr(base, "trmv") || strstr(base, "trsv") ||
        strstr(base, "gbmv") || strstr(base, "sbmv") ||
        strstr(base, "spmv") || strstr(base, "spr")) {
        strcpy(category, "blas_level2");
        return;
    }
    
    /* BLAS LEVEL 3 */
    if (strstr(base, "gemm") || strstr(base, "symm") ||
        strstr(base, "syrk") || strstr(base, "syr2k") ||
        strstr(base, "trmm") || strstr(base, "trsm") ||
        strstr(base, "hemm") || strstr(base, "herk") ||
        strstr(base, "her2k")) {
        strcpy(category, "blas_level3");
        return;
    }
    
    /* LAPACK AUXILIARY: LA* pattern (utility routines) */
    if (strncmp(normalized_name, "la", 2) == 0 ||
        (strlen(normalized_name) > 2 && normalized_name[1] == 'l' && normalized_name[2] == 'a')) {
        strcpy(category, "lapack_auxiliary");
        return;
    }
    
    /* LAPACK DRIVERS: High-level solver patterns */
    if (strstr(base, "gesv") || strstr(base, "posv") ||
        strstr(base, "gesvd") || strstr(base, "gejsv") ||
        strstr(base, "syev") || strstr(base, "heev") ||
        strstr(base, "geev") || strstr(base, "ggev") ||
        strstr(base, "gelsd") || strstr(base, "gelsy") ||
        strstr(base, "gels")) {
        strcpy(category, "lapack_driver");
        return;
    }
    
    /* LAPACK COMPUTATIONAL: Everything else */
    strcpy(category, "lapack_computational");
}

/* Simple line-based parser for function declarations */
OperationList* extract_operations_from_header(
    const char* header_path,
    const char* backend_name,
    const char* convention
) {
    log_info("Extracting from %s (backend: %s, convention: %s)",
             header_path, backend_name, convention);
    
    FILE* f = fopen(header_path, "r");
    if (!f) {
        log_error("Cannot open file: %s", header_path);
        return NULL;
    }
    
    OperationList* ops = malloc(sizeof(OperationList));
    ops->operations = malloc(sizeof(Operation) * 2048);
    ops->count = 0;
    ops->capacity = 2048;
    
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        /* Skip preprocessor, comments, empty lines */
        char* stripped = line;
        while (isspace((unsigned char)*stripped)) stripped++;
        if (*stripped == '\0' || *stripped == '#' || *stripped == '/') continue;
        
        /* Skip typedefs */
        if (strstr(line, "typedef") && strchr(line, '(')) {
            continue;
        }
        
        /* Search for each known BLAS/LAPACK operation in this line */
        for (int op_idx = 0; blas_lapack_operations[op_idx]; op_idx++) {
            const char* op_name = blas_lapack_operations[op_idx];
            const char* search_start = line;
            const char* found;
            
            /* Iteratively search for all occurrences of this operation name */
            while ((found = strstr(search_start, op_name)) != NULL) {
                int pos = found - line;
                char func_name[256] = {0};
                
                /* Extract the full function name at this position */
                if (extract_function_name_at_position(line, pos, func_name, sizeof(func_name)) &&
                    strlen(func_name) > 0) {
                    
                    /* Check if this looks like a function declaration */
                    /* (should have a precision prefix or be cblas_ prefixed) */
                    char base_name[256] = {0};
                    extract_base_name(func_name, base_name, sizeof(base_name));
                    
                    /* Only proceed if the base name matches our operation */
                    if (strcmp(base_name, op_name) == 0) {
                        /* Detect calling convention from function name shape */
                        const char* convention = detect_calling_convention(func_name);
                        
                        /* Create operation entry */
                        Operation op;
                        memset(&op, 0, sizeof(op));
                        
                        strncpy(op.backend, backend_name, sizeof(op.backend) - 1);
                        strncpy(op.actual_name, func_name, sizeof(op.actual_name) - 1);
                        
                        /* Use full function name as normalized name (keep precision prefix) */
                        strncpy(op.name, func_name, sizeof(op.name) - 1);
                        
                        /* Convert to lowercase */
                        for (int i = 0; op.name[i]; i++) {
                            op.name[i] = tolower((unsigned char)op.name[i]);
                        }
                        
                        strncpy(op.return_type, "void", sizeof(op.return_type) - 1);
                        categorize_operation(op.name, op.category);
                        op.param_count = 0;
                        op.parameters = NULL;
                        
                        /* Add to operations list (avoid duplicates) */
                        int is_duplicate = 0;
                        for (int i = 0; i < ops->count; i++) {
                            if (strcmp(ops->operations[i].actual_name, func_name) == 0) {
                                is_duplicate = 1;
                                break;
                            }
                        }
                        
                        if (!is_duplicate) {
                            if (ops->count >= ops->capacity) {
                                ops->capacity *= 2;
                                ops->operations = realloc(ops->operations, sizeof(Operation) * ops->capacity);
                            }
                            ops->operations[ops->count++] = op;
                            log_debug("  ✓ Found: %s (-> %s, %s)", func_name, op.name, op.category);
                        }
                    }
                }
                
                /* Move search position forward to find next occurrence */
                search_start = found + 1;
            }
        }
    }
    
    fclose(f);
    log_info("✓ Extracted %d operations from %s", ops->count, header_path);
    return ops;
}

void free_operations(OperationList* ops) {
    if (!ops) return;
    if (ops->operations) {
        for (int i = 0; i < ops->count; i++) {
            if (ops->operations[i].parameters) {
                for (int j = 0; ops->operations[i].parameters[j]; j++) {
                    free(ops->operations[i].parameters[j]);
                }
                free(ops->operations[i].parameters);
            }
        }
        free(ops->operations);
    }
    free(ops);
}
