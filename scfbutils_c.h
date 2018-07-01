/*
 * Header file for all functions and structures defined in scfbutils_c.c. Must
 * be available for scfbutils.d.pxd file, where all relevant contents must be
 * redclared in appropriate Cython format
 *
 */

double *template_vals(double *f_vals, int num_vals, double f0, double sigma, int num_h);
