cmake_minimum_required(VERSION 3.10)

# ============================================================================
# Build the fb_codegen code generation tool
# ============================================================================

add_executable(fb_codegen
    src/main.c
    src/extractor.c
    src/generator.c
    src/library_loader.c
    src/logging.c
)

# Set C standard
set_target_properties(fb_codegen PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)

# On Unix, link with libm and dl
if(UNIX)
    target_link_libraries(fb_codegen PRIVATE m dl)
endif()

# Platform-specific definitions
if(MSVC)
    target_compile_definitions(fb_codegen PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

message(STATUS "✓ fb_codegen build configured")
