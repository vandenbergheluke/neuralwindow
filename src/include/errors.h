#ifndef ERROR_H
#define ERROR_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_MEMORY_ALLOCATED(x) {\
            if (x == NULL) {\
                fprintf(stderr, "error:%s:%s:%d:%s:failed to allocate memory.\n", __FILE__, __FUNCTION__, __LINE__, strerror(errno));\
                exit(EXIT_FAILURE);\
            }\
        }
#endif
