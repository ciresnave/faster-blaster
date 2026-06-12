/**
 * Simple logging utilities for code generation tool
 */

#ifndef LOGGING_H
#define LOGGING_H

#include <stdio.h>

#define LOG_LEVEL_DEBUG 0
#define LOG_LEVEL_INFO  1
#define LOG_LEVEL_WARN  2
#define LOG_LEVEL_ERROR 3

extern int g_log_level;

void log_debug(const char* fmt, ...);
void log_info(const char* fmt, ...);
void log_warn(const char* fmt, ...);
void log_error(const char* fmt, ...);

#endif
