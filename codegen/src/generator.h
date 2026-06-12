/**
 * Code generation: rendering templates and outputting wrapper files
 */

#ifndef GENERATOR_H
#define GENERATOR_H

#include "extractor.h"

typedef struct {
    const char* backend;
    OperationList* operations;
    const char* output_dir;
    const char* templates_dir;
} GenerationConfig;

/**
 * Generate wrapper files from operations
 */
int generate_wrappers(GenerationConfig* config);

/**
 * Append generated header filename to master include file
 */
int append_to_master_include(
    const char* output_dir,
    const char* master_file,
    const char* header_filename
);

#endif
