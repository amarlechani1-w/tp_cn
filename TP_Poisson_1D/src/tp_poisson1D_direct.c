/******************************************/
/* tp2_poisson1D_direct.c                 */
/* This file contains the main function   */
/* to solve the Poisson 1D problem        */
/* using direct methods (LU factorization)*/
/******************************************/
#include "lib_poisson1D.h"
#include <time.h>

#define TRF 0  /* Use LAPACK dgbtrf for LU factorization */
#define TRI 1  /* Use custom tridiagonal LU factorization */
#define SV 2   /* Use LAPACK dgbsv (all-in-one solver) */

/**
 * Main function to solve the 1D Poisson equation -u''(x) = f(x) with Dirichlet BC.
 * 
 * @param argc: Number of command-line arguments
 * @param argv: Array of argument strings
 *              argv[1] (optional): Implementation method (0=TRF, 1=TRI, 2=SV)
 * @return 0 on success
 */
int main(int argc,char *argv[])
/* ** argc: Nombre d'arguments */
/* ** argv: Valeur des arguments */
{
  int ierr;                      /* Error code for various operations */
  int jj;                        /* Loop counter */
  int nbpoints, la;              /* nbpoints: total points, la: interior points */
  int ku, kl, kv, lab;           /* Band matrix parameters: ku/kl=super/sub diagonals, kv=extra space, lab=leading dimension */
  int *ipiv;                     /* Pivot indices for LU factorization */
  int info = 1;                  /* LAPACK info parameter (0=success) */
  int NRHS;                      /* Number of right-hand sides */
  int IMPLEM = 0;                /* Implementation method (TRF, TRI, or SV) */
  double T0, T1;                 /* Boundary conditions: T0 at x=0, T1 at x=1 */
  double *RHS, *EX_SOL, *X;      /* RHS: right-hand side, EX_SOL: exact solution, X: grid points */
  double **AAB;                  /* Unused variable */
  double *AB;                    /* Coefficient matrix in band storage */

  double relres;                 /* Relative forward error */

  if (argc == 2) {
    IMPLEM = atoi(argv[1]);
  } else if (argc > 2) {
    perror("Application takes at most one argument");
    exit(1);
  }

  /* Problem setup */
  NRHS=1;           /* Solving Ax=b with one right-hand side */
  nbpoints=10;      /* Total number of discretization points (including boundaries) */
  la=nbpoints-2;    /* Number of interior points (excluding boundaries) */
  T0=-5.0;          /* Dirichlet boundary condition at x=0 */
  T1=5.0;           /* Dirichlet boundary condition at x=1 */

  printf("--------- Poisson 1D ---------\n\n");
  /* Allocate memory for vectors */
  RHS=(double *) malloc(sizeof(double)*la);      /* Right-hand side vector */
  EX_SOL=(double *) malloc(sizeof(double)*la);   /* Analytical/exact solution */
  X=(double *) malloc(sizeof(double)*la);        /* Grid points */

  /* Initialize the problem: grid, RHS, and exact solution */
  set_grid_points_1D(X, &la);                                /* Create uniform grid */
  set_dense_RHS_DBC_1D(RHS,&la,&T0,&T1);                     /* Set up RHS with BC */
  set_analytical_solution_DBC_1D(EX_SOL, X, &la, &T0, &T1);  /* Compute exact solution */
  
  /* Write initial data to files for visualization */
  write_vec(RHS, &la, "RHS.dat");
  write_vec(EX_SOL, &la, "EX_SOL.dat");
  write_vec(X, &la, "X_grid.dat");

  /* Set up band storage parameters for tridiagonal matrix */
  kv=1;             /* Number of superdiagonals */
  ku=1;             /* Number of superdiagonals in original matrix */
  kl=1;             /* Number of subdiagonals */
  lab=kv+kl+ku+1;   /* Leading dimension of band storage */

  /* Allocate and initialize the coefficient matrix */
  AB = (double *) malloc(sizeof(double)*lab*la);

  set_GB_operator_colMajor_poisson1D(AB, &lab, &la, &kv);
  write_GB_operator_colMajor_poisson1D(AB, &lab, &la, "AB.dat");

  printf("Solution with LAPACK\n");
  ipiv = (int *) calloc(la, sizeof(int));  /* Pivot indices for LU factorization */

  /* LU Factorization using LAPACK's general band factorization */
  if (IMPLEM == TRF) {
    dgbtrf_(&la, &la, &kl, &ku, AB, &lab, ipiv, &info);
  }

  /* LU for tridiagonal matrix (can replace dgbtrf_) - custom implementation */
  if (IMPLEM == TRI) {
    dgbtrftridiag(&la, &la, &kl, &ku, AB, &lab, ipiv, &info);
  }

  /* Back-substitution to solve the system after factorization */
  if (IMPLEM == TRI || IMPLEM == TRF){
    /* Solution (Triangular) - solve using the LU factors */
    if (info==0){
      dgbtrs_("N", &la, &kl, &ku, &NRHS, AB, &lab, ipiv, RHS, &la, &info);
      if (info!=0){printf("\n INFO DGBTRS = %d\n",info);}    
    }else{
      printf("\n INFO = %d\n",info);
    }
  }

  /* Alternative: solve directly using dgbsv (student completion) */
  if (IMPLEM == SV) {
    dgbsv_(&la, &kl, &ku, &NRHS, AB, &lab, ipiv, RHS, &la, &info);
    if (info != 0) {
      printf("\n INFO DGBSV = %d\n", info);
    }
  }

  /* Write results to files */
  write_GB_operator_colMajor_poisson1D(AB, &lab, &la, "LU.dat");  /* LU factors */
  write_xy(RHS, X, &la, "SOL.dat");  /* Solution at grid points (RHS now contains solution) */

  /* Relative forward error - compare numerical solution with exact solution */
  relres = relative_forward_error(RHS, EX_SOL, &la);
  
  printf("\nThe relative forward error is relres = %e\n",relres);

  /* Free allocated memory */
  free(RHS);
  free(EX_SOL);
  free(X);
  free(AB);
  free(ipiv);

  /* BLOC PERFORMANCE */

  char filename[64];
  if (IMPLEM == 0) sprintf(filename, "perf_lapack.dat");
  else if (IMPLEM == 1) sprintf(filename, "perf_manuel.dat");
  else sprintf(filename, "perf_dgbsv.dat");

  FILE *f_bench = fopen(filename, "w");
  if (f_bench == NULL) { perror("Erreur creation fichier perf"); exit(1); }

  int sizes[] = {100, 1000, 10000, 100000, 1000000}; 
  int num_sizes = 5; 
  int k;
  clock_t start, end;
  double cpu_time_used;

  for (k = 0; k < num_sizes; k++) {
      int la_perf = sizes[k];
      
      double *AB_perf = (double *) malloc(sizeof(double) * lab * la_perf);
      double *RHS_perf = (double *) malloc(sizeof(double) * la_perf);
      int *ipiv_perf = (int *) calloc(la_perf, sizeof(int));
      
      set_GB_operator_colMajor_poisson1D(AB_perf, &lab, &la_perf, &kv);
      double BC0_p = -5.0, BC1_p = 5.0;
      set_dense_RHS_DBC_1D(RHS_perf, &la_perf, &BC0_p, &BC1_p);

      start = clock();

      if (IMPLEM == 0) { 
          dgbtrf_(&la_perf, &la_perf, &kl, &ku, AB_perf, &lab, ipiv_perf, &info);
          dgbtrs_("N", &la_perf, &kl, &ku, &NRHS, AB_perf, &lab, ipiv_perf, RHS_perf, &la_perf, &info);
      } else if (IMPLEM == 1) { 
          dgbtrftridiag(&la_perf, &la_perf, &kl, &ku, AB_perf, &lab, ipiv_perf, &info);
          dgbtrs_("N", &la_perf, &kl, &ku, &NRHS, AB_perf, &lab, ipiv_perf, RHS_perf, &la_perf, &info);
      } else { 
          dgbsv_(&la_perf, &kl, &ku, &NRHS, AB_perf, &lab, ipiv_perf, RHS_perf, &la_perf, &info);
      }

      end = clock();
      cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

      fprintf(f_bench, "%d %f\n", la_perf, cpu_time_used);

      free(AB_perf);
      free(RHS_perf);
      free(ipiv_perf);
  }

  fclose(f_bench);

  printf("\n\n--------- End -----------\n");
  return 0;
}
