#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>

#include <xmmintrin.h>

#define TEST_W 4096
#define TEST_H 4096

/* provide the implementations of naive_transpose,
 * sse_transpose, sse_prefetch_transpose
 */

#include "impl.c"

typedef void (*func_t)(int *src, int *dst, int w, int h);
typedef struct __Object Object;

struct __Object {
    char mode[15];
    func_t trans_func;
};

int init_naive(Object **self)
{
    if((*self = malloc(sizeof(Object))) == NULL) return -1;
    strcpy((*self)->mode, "naive");
    (*self)->trans_func = naive_transpose;
    return 0;
}

int init_sse(Object **self)
{
    if((*self = malloc(sizeof(Object))) == NULL) return -1;
    strcpy((*self)->mode, "sse");
    (*self)->trans_func = sse_transpose;
    return 0;
}

int init_sse_prefetch(Object **self)
{
    if((*self = malloc(sizeof(Object))) == NULL) return -1;
    strcpy((*self)->mode, "see_prefetch");
    (*self)->trans_func = sse_prefetch_transpose;
    return 0;
}

typedef struct __Pool {
    int *nextAvail;
    int *end;
} Pool;

Pool *pool_create(size_t size)
{
    Pool *poolPtr = (Pool *) malloc(size * sizeof(int) + sizeof(Pool));
    if (poolPtr == NULL) printf("Pool Create Failed.\n");
    poolPtr->nextAvail = (int *)&poolPtr[1];
    poolPtr->end = poolPtr->nextAvail + size;

    return poolPtr;
}

void pool_free(Pool *poolPtr)
{
    free(poolPtr);
}

int pool_available(Pool *poolPtr, size_t size)
{
    /* Yet consider alignment */
    return ((size_t)(poolPtr->end - poolPtr->nextAvail)) < size ? 0 : 1;
}

void *pool_alloc(Pool *poolPtr, size_t size)
{
    int check;
    if ((check = pool_available(poolPtr, size)) == 0) return NULL;
    void *mem = (void *) poolPtr->nextAvail;
    poolPtr->nextAvail += size;

    return mem;
}

static long diff_in_us(struct timespec t1, struct timespec t2)
{
    struct timespec diff;
    if (t2.tv_nsec-t1.tv_nsec < 0) {
        diff.tv_sec  = t2.tv_sec - t1.tv_sec - 1;
        diff.tv_nsec = t2.tv_nsec - t1.tv_nsec + 1000000000;
    } else {
        diff.tv_sec  = t2.tv_sec - t1.tv_sec;
        diff.tv_nsec = t2.tv_nsec - t1.tv_nsec;
    }
    return (diff.tv_sec * 1000000.0 + diff.tv_nsec / 1000.0);
}

int main()
{
    /* verify the result of 4x4 matrix */
    int testin[16] = { 0, 1,  2,  3,  4,  5,  6,  7,
                       8, 9, 10, 11, 12, 13, 14, 15
                     };
    int testout[16];
    int expected[16] = { 0, 4,  8, 12, 1, 5,  9, 13,
                         2, 6, 10, 14, 3, 7, 11, 15
                       };

    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++)
            printf(" %2d", testin[y * 4 + x]);
        printf("\n");
    }
    printf("\n");

    /* create a object and init it with corresponding define */
    Object *interface = NULL;
#ifdef NAIVE
    if (init_naive(&interface) == -1) {
        printf("init naive error.\n");
        return -1;
    }
#elif defined(SSE_PREFETCH)
    if (init_sse_prefetch(&interface) == -1) {
        printf("init sse_prefetch error.\n");
        return -1;
    }
#else
    if (init_sse(&interface) == -1) {
        printf("init sse error.\n");
        return -1;
    }
#endif
    interface->trans_func(testin, testout, 4, 4);

    /* Original transpose
     * sse_transpose(testin, testout, 4, 4);*/

    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++)
            printf(" %2d", testout[y * 4 + x]);
        printf("\n");
    }
    assert(0 == memcmp(testout, expected, 16 * sizeof(int)) &&
           "Verification fails");

    struct timespec start, end;
    /*  Original malloc
     *  int *src  = (int *) malloc(sizeof(int) * TEST_W * TEST_H);
     *  int *out0 = (int *) malloc(sizeof(int) * TEST_W * TEST_H);
     */

    Pool *p = pool_create(sizeof(int) * TEST_W * TEST_H * 2);
    int *src = (int *)pool_alloc(p, sizeof(int) * TEST_W * TEST_H);
    int *out0 = (int *)pool_alloc(p, sizeof(int) * TEST_W * TEST_H);

    srand(time(NULL));
    for (int y = 0; y < TEST_H; y++)
        for (int x = 0; x < TEST_W; x++)
            *(src + y * TEST_W + x) = rand();

    clock_gettime(CLOCK_REALTIME, &start);
    interface->trans_func(src, out0, TEST_W, TEST_H);
    clock_gettime(CLOCK_REALTIME, &end);
    printf("%s: \t %ld us\n", interface->mode, diff_in_us(start, end));

    /* Origingal free()
     * free(src);
     * free(out0);
     */
    pool_free(p);

    return 0;
}
