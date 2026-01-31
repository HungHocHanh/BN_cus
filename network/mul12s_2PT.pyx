from libc.stdint cimport *

cdef extern:
    uint32_t mul12s_2PT(uint16_t A, uint16_t B)


cpdef int mul(int a,int b):
    return mul12s_2PT(a,b)

