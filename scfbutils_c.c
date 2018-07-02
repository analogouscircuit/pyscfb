/*
 *
 * Definitions of all pure C functions used in the SCFB implementation.  Note
 * that all these functions must be declared in both scfbutils_c.h and
 * scfbutils_d.pyd.  The wrapper is defined in the main Cython file,
 * scfbutils.pyx
 *
 */

#include <math.h>
#include <stdlib.h>

double *template_vals(double *f_vals, int num_vals, double f0, double sigma, int num_h)
{
	double *t_vals = malloc(num_vals * sizeof(double));
	double factor;
	int lim = num_h+1;
	
	for(int k = 0; k < num_vals; k++)
	{
		t_vals[k] = 0.0;
		for(int p = 1; p < lim; p++)
		{
			factor = p == 1 ? 1.0 : 0.8;
			//t_vals[k] += exp( - pow(f_vals[k] - p*f0, 2)/( 2 * pow(sigma, 2)) );
			t_vals[k] += factor * exp( - pow(f_vals[k] - p*f0, 2)/( 2 * pow(0.03*p*f0, 2)) );
		}
	}	
	
	return t_vals;
}
