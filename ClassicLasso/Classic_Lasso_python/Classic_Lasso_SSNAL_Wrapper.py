import numpy as np
from Classic_Lasso_SSNAL import *

def classic_lasso_ssnal_wrapper(Ainput, b, n, lam, options, y=None, xi=None, x=None):
    """
    A Python translation of the MATLAB Classic_Lasso_SSNAL_Wrapper function.

    Solves the Lasso problem
        (P)   min 1/2 ||A x - b||^2 + lam * ||x||_1
    either by directly calling classic_lasso_ssnal when m <= n or n > 1e4,
    or by using a QR-based reduction when m > n and n <= 10_000.

    Parameters
    ----------
    Ainput : np.ndarray or dict
        Either a 2D NumPy array A of shape (m, n) or a dict containing key 'A'.
    b : np.ndarray
        A 1D NumPy array of length m (right-hand side vector).
    n : int
        Number of features/columns in A (passed explicitly as in the MATLAB version).
    lam : float
        The regularization parameter λ.
    options : dict
        A dictionary of options; this function may set options['orgojbconst'] when using QR.
    y : np.ndarray, optional
        Initial guess for the dual variable y (length n). Defaults to zeros(n).
    xi : np.ndarray, optional
        Initial guess for the dual variable ξ (length m). Defaults to zeros(m).
    x : np.ndarray, optional
        Initial guess for the primal variable x (length n). Defaults to zeros(n).

    Returns
    -------
    obj : float
        The objective value at the solution.
    y : np.ndarray
        The dual variable y at the solution (length n).
    xi : np.ndarray
        The dual variable ξ at the solution (length m).
    x : np.ndarray
        The primal solution x (length n).
    info : dict or custom
        A dictionary or custom info object returned by classic_lasso_ssnal.
    runhist : list or custom
        A list or custom run-history object returned by classic_lasso_ssnal.
    """
    # Initial assumption: use QR-based reduction if possible
    isQR = True
    QRtol = 1e-8

    # If Ainput is a dict, try to extract 'A'; otherwise, assume it's already the matrix
    if isinstance(Ainput, dict):
        if 'A' in Ainput:
            A = Ainput['A']
        else:
            # Struct without 'A' → cannot do QR-based reduction
            isQR = False
    else:
        A = Ainput

    m = b.shape[0]

    # Decide whether to use QR or call original SSNAL directly
    if m > n:
        if n > 1e4:
            isQR = False
        # else: keep isQR = True
    else:
        # If m <= n, do not use QR
        isQR = False

    if not isQR:
        print("\nCalling original SSNAL to solve the problem...")

        # Initialize y, xi, x if they were not provided
        if x is None:
            x = np.zeros(n, dtype=float)
        if y is None:
            y = np.zeros(n, dtype=float)
        if xi is None:
            xi = np.zeros(m, dtype=float)

        # Note: The function classic_lasso_ssnal must be defined elsewhere in Python.
        obj, y, xi, x, info, runhist = classic_lasso_ssnal(Ainput, b, n, lam, options, y, xi, x)
        return obj, y, xi, x, info, runhist

    else:
        print(f"\nCase m = {m}, n = {n} (m > n), using QR decomposition to rewrite the problem.")

        # Economy-size QR factorization (A is m × n with m > n)
        # NumPy's 'reduced' mode yields Q of shape (m, n) and R of shape (n, n).
        Q, R = np.linalg.qr(A, mode='reduced')

        # If the bottom-right diagonal of R is nearly zero, drop near-zero rows of R
        if abs(R[n - 1, n - 1]) < QRtol:
            diagR = np.abs(np.diag(R))
            idx = diagR >= QRtol
            Q = Q[:, idx]
            R = R[idx, :]

        # Update dimensions after potential rank reduction
        m_reduced, n_reduced = R.shape
        print(f"\nNew data constructed with m = {m_reduced}, n = {n_reduced} (after dropping near-zero rows).")

        # Form the reduced b: b_new = Q^T * b
        borg = b.copy()
        b_new = Q.T.dot(borg)

        # Compute constant term: 0.5*(||borg||^2 - ||b_new||^2)
        const = 0.5 * (np.linalg.norm(borg)**2 - np.linalg.norm(b_new)**2)
        options['orgojbconst'] = const

        # Call classic_lasso_ssnal on the reduced problem: R x = b_new
        # The reduced problem dimension is (m_reduced, n_reduced)
        obj, y, xi, x, info, runhist = classic_lasso_ssnal(R, b_new, n_reduced, lam, options)
        return obj, y, xi, x, info, runhist
