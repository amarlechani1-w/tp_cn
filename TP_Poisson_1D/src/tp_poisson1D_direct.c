/******************************************/
/* tp2_poisson1D_direct.c                 */
/* This file contains the main function   */
/* to solve the Poisson 1D problem        */
/* using direct methods (LU factorization)*/
/******************************************/
#include "lib_poisson1D.h"

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
  int ku, kl, kv, lab;           /* Band matrix parameters */
  int *ipiv;                     /* Pivot indices */
  int info = 1;                  /* LAPACK info */
  int NRHS;                      /* Number of RHS */
  int IMPLEM = 0;                /* Implementation method */
  double T0, T1;                 /* Boundary conditions */
  double *RHS, *EX_SOL, *X;      /* Vectors */
  double **AAB;                  /* Unused */
  double *AB;                    /* Band matrix */

  double relres;                 /* Relative forward error */

  if (argc == 2) {
    IMPLEM = atoi(argv[1]);
  } else if (argc > 2) {
    perror("Application takes at most one argument");
    exit(1);
  }

  /* Problem setup */
  NRHS=1;
  nbpoints=10;
  la=nbpoints-2;
  T0=-5.0;
  T1=5.0;

  printf("--------- Poisson 1D ---------\n\n");

  RHS=(double *) malloc(sizeof(double)*la);
  EX_SOL=(double *) malloc(sizeof(double)*la);
  X=(double *) malloc(sizeof(double)*la);

  /* Initialize the problem */
  set_grid_points_1D(X, &la);
  set_dense_RHS_DBC_1D(RHS,&la,&T0,&T1);
  set_analytical_solution_DBC_1D(EX_SOL, X, &la, &T0, &T1);
  
  write_vec(RHS, &la, "RHS.dat");
  write_vec(EX_SOL, &la, "EX_SOL.dat");
  write_vec(X, &la, "X_grid.dat");

  /* Band matrix parameters */
  kv=1;
  ku=1;
  kl=1;
  lab=kv+kl+ku+1;

  AB = (double *) malloc(sizeof(double)*lab*la);

  set_GB_operator_colMajor_poisson1D(AB, &lab, &la, &kv);
  write_GB_operator_colMajor_poisson1D(AB, &lab, &la, "AB.dat");

  printf("Solution with LAPACK\n");
  ipiv = (int *) calloc(la, sizeof(int));

  /* LU Factorization */
  if (IMPLEM == TRF) {
    dgbtrf_(&la, &la, &kl, &ku, AB, &lab, ipiv, &info);
  }

  /* LU for tridiagonal matrix */
  if (IMPLEM == TRI) {
    dgbtrftridiag(&la, &la, &kl, &ku, AB, &lab, ipiv, &info);
  }

  /* Back substitution */
  if (IMPLEM == TRI || IMPLEM == TRF){
    if (info==0){
      dgbtrs_("N", &la, &kl, &ku, &NRHS, AB, &lab, ipiv, RHS, &la, &info);
      if (info!=0){printf("\n INFO DGBTRS = %d\n",info);}
    }else{
      printf("\n INFO = %d\n",info);
    }
  }

  /* Alternative: solve directly using dgbsv */
  if (IMPLEM == SV) {
    dgbsv_(&la, &kl, &ku, &NRHS, AB, &lab, ipiv, RHS, &la, &info);
    if (info != 0) {
      printf("\n INFO DGBSV = %d\n", info);
    }
  }

  /* Write results */
  write_GB_operator_colMajor_poisson1D(AB, &lab, &la, "LU.dat");
  write_xy(RHS, X, &la, "SOL.dat");

  /* Relative forward error */
  relres = relative_forward_error(RHS, EX_SOL, &la);
  printf("\nThe relative forward error is relres = %e\n",relres);

  /* Free memory */
  free(RHS);
  free(EX_SOL);
  free(X);
  free(AB);
  free(ipiv);

  printf("\n\n--------- End -----------\n");
}
