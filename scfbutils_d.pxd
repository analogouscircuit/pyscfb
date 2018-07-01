'''
The declaration file for any pure C functions or structures defined in
scfbutils_c.c and declared (in C) in scfbutils_c.h.
'''

cdef extern from "scfbutils_c.h":
    double *template_vals(double *f_vals, int num_vals, double f0, 
                          double sigma, int num_h)
