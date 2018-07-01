# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

cdef unsigned int INT_SIZE = 4

from cython cimport view
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.stdio cimport FILE, fopen, fread, fseek, fclose, SEEK_SET
import numpy as np

cdef inline unsigned int le_to_cpu_i(unsigned char *buf):
    return buf[0] | (buf[1] << 8) | (buf[2] << 16) | (buf[3] << 24)

cpdef list read_XXXX(unicode file_name, bint stats=False):
    cdef unsigned char buf[12]
    cdef char *word_buf
    cdef float *vec_buf
    cdef unsigned int header[3]
    cdef unsigned int vocab, dim, length
    cdef unsigned int i, j
    cdef unsigned int line_length, vec_length
    cdef unsigned int offset = INT_SIZE * 3
    cdef float[:, :] vectors
    cdef char **words
    cdef FILE *f
    cdef void *data
    f = fopen(file_name.encode('utf-8'), b'rb')
    if f == NULL:
        raise IOError
    fread(buf, INT_SIZE, 3, f)
    for i in range(3):
        header[i] = le_to_cpu_i(&buf[i * INT_SIZE])
    vocab = header[0]
    dim = header[1]
    length = header[2]
    cdef view.array vectors_arr = view.array(shape=(vocab, dim), itemsize=sizeof(float), format="f")
    vectors = vectors_arr
    vec_length = dim * sizeof(float)
    words = <char **>malloc(vocab * sizeof(char*))
    line_length = length + dim * INT_SIZE
    # for i in prange(vocab, nogil=True):
    for i in range(vocab):
        start = offset + i * line_length
        word_buf = <char *>malloc(length + 1 * sizeof(unsigned char))
        word_buf[length] = b'\0'
        vec_buf = <float *>malloc(vec_length * sizeof(float))
        fseek(f, offset + i * line_length, SEEK_SET)
        fread(word_buf, sizeof(unsigned char), length, f)
        fread(vec_buf, sizeof(float), vec_length, f)
        words[i] = word_buf
        for j in range(dim):
            vectors[i, j] = vec_buf[j]
        for j in range(length + 1):
            if word_buf[j] == b' ':
                word_buf[j] = b'\0'
                break
    # test = []
    # for i in range(vocab):
    #     test.append(words[i])
    # return test
