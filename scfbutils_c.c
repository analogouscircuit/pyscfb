/*
 * File: 	scfbutils_c.c
 * ----------------------
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


/*
 * Function: 	template_vals
 * --------------------------
 * This produces all the template values, given a set of frequencies at
 * which to calculate. Primarily useful for visualizing the templates, but
 * could also be used if the input is non-sparse and one wishes to do a
 * complete inner-product calculation.
 */

double *template_vals_c(double *f_vals, int num_vals, double f0, double sigma, int num_h)
{

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


/*
 * Function: 	template_adapt_c
 * -----------------------------
 *  The main signal processing for the template adaptation. Operates on a linked
 *  list containing a variable number of frequency estimates for each moment in
 *  time. Performs a gradient ascent calculation to determine the appropriate
 *  adaptation behavior. Also calculates the instantaneous strength (how well
 *  the template matches the input).
 */

fs_struct template_adapt_c(f_list **f_estimates, int list_len, double f0, 
						   double mu, int num_h, double sigma)
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
		for(n = 0; n < f_estimates[k]->count; n++) {
			freq = fl_by_idx(n, f_estimates[k]);
			for(p = 1; p < num_h+1; p++) {
				factor = p == 1 ? 1.0 : 0.8;
				temp = factor * exp( - pow(freq - p*f[k],2)/(2*pow(sigma, 2)));
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

/*
 * Functions for performing the winner-take-all (WTA) network calculations. This
 * is just an implementation of the fourth-order Runge-Kutta method for a system
 * of equations. The 'right hand side' used for this system in contained in the
 * function rhs below.
 */

double *wta_net_c(double *E, double *k, int num_n, int sig_len, double dt,
				 double *tau, double M, double N, double sigma)
{
	double *x;
	double x_input[num_n];
	double E_input[num_n];
	double *x_old; 	// initial conditions
	double K1[num_n], K2[num_n], K3[num_n], K4[num_n];
	int i, j, t;
	double sum;
	double hdt = dt/2.0;

	x = calloc(num_n * sig_len, sizeof(double));
	x_old = calloc(num_n, sizeof(double));

	printf("E_input[0]: %f\n",   E[0*sig_len + 10]);
    printf("E_input[1]: %f\n",   E[1*sig_len + 10]);
	printf("E_input[2]: %f\n",   E[2*sig_len + 10]);  
    printf("E_input[3]: %f\n",   E[3*sig_len + 10]);
	printf("E_input[4]: %f\n\n", E[4*sig_len + 10]);

	for(t = 0; t < sig_len-1; t++) {
		// Find K1 vals
		for(i = 0; i < num_n; i++) {
			x_input[i] = x[i*sig_len + t];
			sum = 0;
			for(j = 0; j < num_n; j++){
				if(j!=i) sum += k[j*num_n+i]*x[j*sig_len + t];
			}
			E_input[i] = plus(E[i*sig_len + t] - sum);
		}
		rhs(x_input, E_input, K1, num_n, k, tau, M, N, sigma);
		
		// Find K2 vals
		for(i = 0; i < num_n; i++) {
			x_input[i] = x[i*sig_len + t] + hdt*K1[i];
		}
		rhs(x_input, E_input, K2, num_n, k, tau, M, N, sigma);

		// Find K3 vals
		for(i = 0; i < num_n; i++) {
			x_input[i] = x[i+sig_len + t] + hdt*K2[i];
		}
		rhs(x_input, E_input, K3, num_n, k, tau, M, N, sigma);

		// Find K4 vals
		for(i = 0; i < num_n; i++) {
			x_input[i] = x[i*sig_len+t] + dt*K3[i];
		}
		rhs(x_input, E_input, K4, num_n, k, tau, M, N, sigma);	

		// Finally calculate update
		for(i = 0; i < num_n; i++) {
			x[i*sig_len + (t+1)] = x_old[i] + (dt/6.0)*(K1[i] + 2*K2[i] + 2*K3[i] + K4[i]);
			x_old[i] = x[i*sig_len + (t+1)];
		}
	}
	return x;
}

void rhs(double *x, double *E, double *out, int num_n, double *k, double *tau,
			double M, double N, double sigma)
{
	for(int i = 0; i < num_n; i++) {
		out[i] = (-x[i] + naka_rushton(E[i], M, N, sigma))/tau[i];	
	}
}

double naka_rushton(double p, double M, double N, double sigma)
{
	double p_N = pow(p, N);
	return M * (p_N/(pow(sigma, N) + p_N));
}

inline double plus(double a)
{
	if(a > 0) {
		return a;
	} else {
		return 0;
	}
}

/* 
 * Functions for linked lists. This allows having a variable number of frequency
 * estimates for each moment in time. These are all basic utility functions.
 */

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
