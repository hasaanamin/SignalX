/* MSVC compatibility for the POSIX bzero call in the unchanged reference. */
#include <string.h>
#define bzero(pointer, length) memset((pointer), 0, (length))
