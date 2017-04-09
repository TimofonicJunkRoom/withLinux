#include <luaT.h>
#include <TH/TH.h>

#define real float

typedef struct THRealStorage
{
    real *data;
    ptrdiff_t size;
    int refcount;
    char flag;
    THAllocator *allocator;
    void *allocatorContext;
} THRealStorage;

typedef struct THRealTensor
{
    long *size;
    long *stride;
    int nDimension;

    THRealStorage *storage;
    ptrdiff_t storageOffset;
    int refcount;

    char flag;

} THRealTensor;

TH_API void THTensor_(myadd)(THTensor *r_, THTensor *t, real value);
