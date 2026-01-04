/**********************************************/
/* lib_poisson1D.c                            */
/* Numerical library developed to solve 1D    */ 
/* Poisson problem (Heat equation)            */
/**********************************************/
#include "lib_poisson1D.h"

void eig_poisson1D(double* eigval, int *la){
  // TODO: Compute all eigenvalues for the 1D Poisson operator
  int k;
  double h = (1.0 / ((double)(*la) + 1.0));
  for (k = 1; k <= *la; ++k){
    double arg = (double)k * M_PI * h;
    eigval[k - 1] = 2.0 - 2.0 * cos(arg);
  }
}

double eigmax_poisson1D(int *la){
  // TODO: Compute and return the maximum eigenvalue for the 1D Poisson operator
  double h = 1.0 / ((double)(*la) + 1.0);
  double arg = (double)(*la) * M_PI * h;
  return 2.0 - 2.0 * cos(arg);
}

double eigmin_poisson1D(int *la){
  // TODO: Compute and return the minimum eigenvalue for the 1D Poisson operator
  double h = 1.0 / ((double)(*la) + 1.0);
  double arg = M_PI * h;
  return 2.0 - 2.0 * cos(arg);
}

double richardson_alpha_opt(int *la){
  // TODO: Compute alpha_opt
  return 2.0 / (eigmax_poisson1D(la) + eigmin_poisson1D(la));
}

/**
 * Solve linear system Ax=b using Richardson iteration with fixed relaxation parameter alpha.
 * The iteration is: x^(k+1) = x^(k) + alpha*(b - A*x^(k))
 * Stops when ||b - A*x^(k)||_2  / ||b||_2 < tol or when reaching maxit iterations.
 */
void richardson_alpha(double *AB, double *RHS, double *X, double *alpha_rich, int *lab, int *la,int *ku, int*kl, double *tol, int *maxit, double *resvec, int *nbite){
  // TODO: Implement Richardson iteration
  double *r = (double *)malloc((*la) * sizeof(double));
  double norm_residual, norm_b;
  double res = *tol + 1.0;
  int k = 0;

  norm_b = cblas_dnrm2(*la, RHS, 1);

  while (res > *tol && k < *maxit) {
    cblas_dcopy(*la, RHS, 1, r, 1);

    cblas_dgbmv(CblasColMajor, CblasNoTrans, *la, *la, *kl, *ku,
                -1.0, AB, *lab, X, 1, 1.0, r, 1);

    norm_residual = cblas_dnrm2(*la, r, 1);
    
    if (norm_b != 0.0) {
      res = norm_residual / norm_b;
    } else {
      res = norm_residual;
    }
    
    resvec[k] = res;

    if (res < *tol) break;

    cblas_daxpy(*la, *alpha_rich, r, 1, X, 1);

    k++;
  }

  *nbite = k;
  free(r);
}

/**
 * Extract MB for Jacobi method from tridiagonal matrix.
 * Such as the Jacobi iterative process is: x^(k+1) = x^(k) + D^(-1)*(b - A*x^(k))
 */
void extract_MB_jacobi_tridiag(double *AB, double *MB, int *lab, int *la,int *ku, int*kl, int *kv){
  // TODO: Extract diagonal elements from AB and store in MB
  int j;
  int kv_loc = (*lab) - (*kl) - (*ku) - 1; // Deduce kv from lab

  for (j = 0; j < (*lab) * (*la); j++) MB[j] = 0.0;

  for (j = 0; j < *la; j++) {
    MB[(kv_loc + *ku) + j * (*lab)] = AB[(kv_loc + *ku) + j * (*lab)];
  }
}

/**
 * Extract MB for Gauss-Seidel method from tridiagonal matrix.
 * Such as the Gauss-Seidel iterative process is: x^(k+1) = x^(k) + (D-E)^(-1)*(b - A*x^(k))
 */
void extract_MB_gauss_seidel_tridiag(double *AB, double *MB, int *lab, int *la,int *ku, int*kl, int *kv){
  // TODO: Extract diagonal and lower diagonal from AB
  int j;
  int kv_loc = (*lab) - (*kl) - (*ku) - 1;

  for (j = 0; j < (*lab) * (*la); j++) MB[j] = 0.0;

  for (j = 0; j < *la; j++) {
    MB[(kv_loc + *ku) + j * (*lab)] = AB[(kv_loc + *ku) + j * (*lab)];

    if (j < *la - 1) {
      MB[(kv_loc + *ku + 1) + j * (*lab)] = AB[(kv_loc + *ku + 1) + j * (*lab)];
    }
  }
}

/**
 * Solve linear system Ax=b using preconditioned Richardson iteration.
 * The iteration is: x^(k+1) = x^(k) + M^(-1)*(b - A*x^(k))
 * where M is either D for Jacobi or (D-E) for Gauss-Seidel.
 * Stops when ||b - A*x^(k)||_2  / ||b||_2 < tol or when reaching maxit iterations.
 */
void richardson_MB(double *AB, double *RHS, double *X, double *MB, int *lab, int *la,int *ku, int*kl, double *tol, int *maxit, double *resvec, int *nbite){
  // TODO: Implement Richardson iterative method
  double *r = (double *)malloc((*la) * sizeof(double));
  double norm_residual, norm_b;
  double res = *tol + 1.0;
  int k = 0;
  int i;
  
  int kv_loc = (*lab) - (*kl) - (*ku) - 1;
  int idx_diag = kv_loc + *ku;
  int idx_sub  = kv_loc + *ku + 1;
  int is_gs = 0; 

  for (i = 0; i < *la - 1; i++) {
    if (fabs(MB[idx_sub + i * (*lab)]) > 0.0) {
      is_gs = 1;
      break;
    }
  }

  norm_b = cblas_dnrm2(*la, RHS, 1);

  while (res > *tol && k < *maxit) {
    cblas_dcopy(*la, RHS, 1, r, 1);
    cblas_dgbmv(CblasColMajor, CblasNoTrans, *la, *la, *kl, *ku,
                -1.0, AB, *lab, X, 1, 1.0, r, 1);

    norm_residual = cblas_dnrm2(*la, r, 1);
    
    if (norm_b != 0.0) {
      res = norm_residual / norm_b;
    } else {
      res = norm_residual;
    }
    resvec[k] = res;

    if (res < *tol) break;
    
    if (is_gs == 0) {
        for (i = 0; i < *la; i++) {
            double diag = MB[idx_diag + i * (*lab)];
            r[i] = r[i] / diag;
        }
    } else {
        for (i = 0; i < *la; i++) {
            double diag = MB[idx_diag + i * (*lab)];
            double sub = 0.0;
            double prev_z = 0.0;

            if (i > 0) {
                sub = MB[idx_sub + (i - 1) * (*lab)];
                prev_z = r[i - 1]; 
            }
            
            r[i] = (r[i] - sub * prev_z) / diag;
        }
    }

    cblas_daxpy(*la, 1.0, r, 1, X, 1);

    k++;
  }

  *nbite = k;
  free(r);
}