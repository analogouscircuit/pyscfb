/*
 * Header file for all functions and structures defined in scfbutils_c.c. Must
 * be available for scfbutils.d.pxd file, where all relevant contents must be
 * redeclared in appropriate Cython format
 *
 */


/* 
 * Structure/Type Declarations
 * ---------------------------
 */

typedef struct f_node_vals_s{
	double f_val;
	double s_val;
} f_node_vals;

typedef struct f_node_s{
	double val;
	struct f_node_s *next;	
} f_node;

typedef struct f_list_s{
	struct f_node_s *head;
	int count;
} f_list;

typedef struct fs_struct_s {
	double *freqs;
	double *strengths;
} fs_struct;



/* 
 * Function Declarations
 * ---------------------
 */

double *template_vals_c(double *f_vals, int num_vals, double f0, double sigma, 
					    int num_h, double *h_size); 

double *template_dvals_c(double *f_vals, int num_vals, double f0, double sigma, 
						int num_h, double *h_size); 

fs_struct template_adapt_c(f_list **f_estimates, int list_len, double f0,
						   double mu, double sigma, int num_h, double *h_size, 
						   double f_lo, double f_hi);

double lin_interp(double x_val, double *x_vals, double *y_vals, int len);

double *wta_net_c(double *E, double *k, int num_n, int sig_len, double dt,
				 double *tau, double M, double N, double sigma);

void rhs(double *x, double *E, double *out, int num_n, double *k, double *tau,
			double M, double N, double sigma);

	
double naka_rushton(double p, double M, double N, double sigma);

double plus(double a);

void fl_push(double freq, f_list *l);

double fl_pop(f_list *l);

void init_f_list(f_list *l);

void free_f_list(f_list *l);

double fl_by_idx(int idx, f_list *l);
