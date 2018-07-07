/*
 *
 * Definitions of all pure C functions used in the SCFB implementation.  Note
 * that all these functions must be declared in both scfbutils_c.h and
 * scfbutils_d.pyd.  The wrapper is defined in the main Cython file,
 * scfbutils.pyx
 *
 */

#include "scfbutils_c.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double *template_vals(double *f_vals, int num_vals, double f0, double sigma, int num_h)
{
	/*
	 * This produces all the template values, given a set of frequencies at
	 * which to calculate. Primarily useful for visualizing the templates, but
	 * could also be used if the input is non-sparse and one wishes to do a
	 * complete inner-product calculation.
	 */

	double *t_vals = malloc(num_vals * sizeof(double));
	double factor;
	int lim = num_h+1;
	
	for(int k = 0; k < num_vals; k++) {
		t_vals[k] = 0.0;
		for(int p = 1; p < lim; p++) {
			factor = p == 1 ? 1.0 : 0.8;
			//t_vals[k] += exp( - pow(f_vals[k] - p*f0, 2)/( 2 * pow(sigma, 2)) );
			t_vals[k] += factor * exp( - pow(f_vals[k] - p*f0, 2)/( 2 * pow(0.03*p*f0, 2)) );
		}
	}	
	return t_vals;
}


fs_struct template_adapt_c(f_list **f_estimates, int list_len, double f0, double mu, int num_h, double sigma)
{
	int k, n, p;
	double *f= malloc(list_len*sizeof(double)); 	// freqs
	double *s= malloc(list_len*sizeof(double)); 	// strengths
	double dJ, temp, factor, freq;
	fs_struct fs;
	f[0] = f0;
	for(k = 0; k < list_len-1; k++) {
		s[k] = 0.0;
		dJ = 0.0;
		// while(f_estimates[k]->count>0) {
		// 	freq = fl_pop(f_estimates[k]);
		// 	for(p = 1; p < num_h+1; p++) {
		// 		factor = p == 1 ? 1.0 : 0.8;
		// 		temp = exp( - pow(freq - p*f[k],2)/(2*pow(sigma, 2)));
		// 		s[k] += temp;
		// 		dJ += temp*(freq - p*f[k])/(p * pow(sigma, 2));
		// 	}
		// }
		for(n = 0; n < f_estimates[k]->count; n++) {
			freq = fl_by_idx(n, f_estimates[k]);
			for(p = 1; p < num_h+1; p++) {
				factor = p == 1 ? 1.0 : 0.8;
				temp = exp( - pow(freq - p*f[k],2)/(2*pow(sigma, 2)));
				s[k] += temp;
				dJ += temp*(freq - p*f[k])/(p * pow(sigma, 2));
			}
		}
		f[k+1] = f[k] + mu*dJ;
		fs.freqs = f;
		fs.strengths = s;
	}
	return fs;
}

void init_f_list(f_list *l)
{
	l->head = NULL;
	l->count = 0;
}

void free_f_list(f_list *l)
{
	while(l->count > 0)
	{
		fl_pop(l);
	}
	free(l);
}

void fl_push(double freq, f_list *l)
{
	f_node *new = malloc(sizeof(f_node));
	new->val = freq;
	new->next = l->head;
	l->head = new;
	l->count += 1;
}

double fl_pop(f_list *l)
{
	if(l->head == NULL) {
		printf("Tried to pop an empty list! Exiting.\n");
		exit(0);
	}
	double val = l->head->val;
	f_node *temp = l->head;
	l->head = l->head->next;
	l->count -= 1;
	free(temp);
	return val;
}

double fl_by_idx(int idx, f_list *l)
{
	if(idx >= l->count) {
		printf("Tried to access non-existent linked list member. Exiting.\n");
		exit(0);
	}
	f_node *temp = l->head;
	for(int p = 0; p < idx; p++) {
		temp = temp->next;
	}
	return temp->val;
}
