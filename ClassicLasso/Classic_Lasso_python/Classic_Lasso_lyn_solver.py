import numpy as np
from scipy import linalg
from scipy.sparse.linalg import cg, LinearOperator

def classic_lasso_linsys_solver(Ainput, rhs, par):
    """
    Python version of the MATLAB Classic_Lasso_linsys_solver.

    Solves either
        (I + sigma * A_pp * A_pp^T) xi = rhs
    or the “projected” variant
        xi = rhs - A_pp * (M^{-1} (A_pp^T rhs)), 

    depending on the size parameters.

    Parameters
    ----------
    Ainput : object with attribute 'A' (numpy.ndarray of shape (m,n))  
        If Ainput.A exists, we use it; otherwise you'd need Amap/ATmap routines.
    rhs : array_like, shape (m,)
        Right‐hand side vector.
    par : object with attributes:
          - rr : boolean array of length n.  True/False masks out columns in A.
          - sigma : positive float.
          - n : integer = number of columns of Ainput.A.

    Returns
    -------
    xi : ndarray, shape (m,)
        The computed solution vector.
    resnrm : float
        Residual norm for the case where we used CG.  Zero if direct solver.
    solve_ok : bool
        True if the CG converged (info==0), or True if direct solver branch was used.
    """
    # Ensure rhs is a flat 1D array
    rhs = np.asarray(rhs).ravel()
    m = len(rhs)

    # Boolean mask for “active” columns in A
    pp = ~par.rr
    Ayes = hasattr(Ainput, 'A')
    solver = 'd_pcg'

    dn = 10000
    sp = np.sum(pp)  # number of active columns

    # ————— Solvers‐selection logic (exactly as in MATLAB) —————
    if (m <= dn) and Ayes:
        if m <= 1000:
            solver = 'd_direct'
        elif sp <= max(0.01 * par.n, dn):
            solver = 'd_direct'

    if (sp <= 0.7 * m) and Ayes and (sp <= dn):
        solver = 'p_direct'

    if ((m > 5e3 and sp >= 200)
        or (m > 2000 and sp > 800)
        or (m > 100 and sp > 1e4)):
        solver = 'd_pcg'
    # ——————————————————————————————————————————————————

    # CASE 1:  “d_pcg” branch
    if solver == 'd_pcg':
        if Ayes:
            AP = Ainput.A[:, pp]  # pick the active columns
            sigma = par.sigma

            # Define as a LinearOperator:  M(x) = x + sigma * AP * (AP^T x)
            def matvec(x):
                return x + sigma * (AP @ (AP.T @ x))

            M_op = LinearOperator((m, m), matvec=matvec)
            xi, info = cg(M_op, rhs)
            solve_ok = (info == 0)

            # Compute the residual norm: || M xi - rhs ||
            resnrm = np.linalg.norm(matvec(xi) - rhs)

        else:
            # The “Amap” version is not implemented in this conversion.
            raise NotImplementedError("d_pcg with Amap is not implemented in this Python translation.")

    # CASE 2:  “d_direct” branch
    elif solver == 'd_direct':
        AP = Ainput.A[:, pp]
        sigma = par.sigma

        # Form sigma * (AP AP^T) once
        sigAPAt = sigma * (AP @ AP.T)

        if m <= 1500:
            M = np.eye(m) + sigAPAt
            xi = np.linalg.solve(M, rhs)
        else:
            # Use Cholesky when m > 1500
            M = np.eye(m) + sigAPAt
            L = np.linalg.cholesky(M)  # L @ L^T = M
            # Solve L L^T xi = rhs via two triangular solves
            y = linalg.solve_triangular(L, rhs, lower=True)
            xi = linalg.solve_triangular(L.T, y, lower=False)

        resnrm = 0.0
        solve_ok = True

    # CASE 3:  “p_direct” branch
    elif solver == 'p_direct':
        AP = Ainput.A[:, pp]
        sigma = par.sigma
        APT = AP.T
        rhstmp = APT @ rhs
        PAtAP = APT @ AP
        sp = PAtAP.shape[0]

        if sp <= 1500:
            M = np.eye(sp) / sigma + PAtAP
            tmp = np.linalg.solve(M, rhstmp)
        else:
            M = np.eye(sp) / sigma + PAtAP
            L = np.linalg.cholesky(M)
            y = linalg.solve_triangular(L, rhstmp, lower=True)
            tmp = linalg.solve_triangular(L.T, y, lower=False)

        xi = rhs - AP @ tmp
        resnrm = 0.0
        solve_ok = True

    else:
        raise ValueError(f"Unrecognized solver string: {solver}")

    return xi, resnrm, solve_ok


# # -------------------------------------------------------------------------
# # Sanity‐check example
# if __name__ == "__main__":
#     """
#     We pick a very small, hand‐verifiable case:

#       A = I_2,   par.rr = [False, False],   sigma = 2.0,
#       rhs = [3.0, -6.0]^T.

#     Then A_pp = I_2, so the linear system solved is
#       (I + 2 * I_2 * I_2^T) xi = rhs  <=>  3 * xi = rhs  <=>  xi = rhs / 3.

#     We expect xi = [1, -2]^T exactly.
#     """

#     class Param: pass

#     # 1) Build Ainput with a 2×2 identity
#     Ainput = Param()
#     Ainput.A = np.array([[3,2],
#                          [1,5]])

#     # 2) Define rhs
#     rhs = np.array([3.0, -6.0])

#     # 3) Build par with rr mask all False, sigma=2, n=2 columns
#     par = Param()
#     par.rr = np.array([False, False])  # no columns excluded
#     par.sigma = 2.0
#     par.n = 2

#     xi, resnrm, solve_ok = classic_lasso_linsys_solver(Ainput, rhs, par)

#     expected_xi = rhs / 3.0
#     print("xi (computed):", xi)
#     print("xi (expected):", expected_xi)
#     print("Residual norm:", resnrm)
#     print("Solve OK flag:", solve_ok)
#     print("Difference norm:", np.linalg.norm(xi - expected_xi))
