#ifndef STRINGS_H
#define STRINGS_H
#include <string.h>
#define bzero(b,len) (memset((b), '\0', (len)), (void) 0)
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#endif
