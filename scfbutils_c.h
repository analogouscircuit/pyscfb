/*
 * Header file for all functions and structures defined in scfbutils_c.c. Must
 * be available for scfbutils.d.pxd file, where all relevant contents must be
 * redclared in appropriate Cython format
 *
 */

double *template_vals(double *f_vals, int num_vals, double f0, double sigma, int num_h);

typedef struct f_node_s{
	double val;
	struct f_node_s *next;	
} f_node;

typedef struct f_list_s{
	struct f_node_s *head;
	int count;
} f_list;


void fl_push(double freq, f_list *list);
double fl_pop(f_list *list);
void init_f_list(f_list *list);
