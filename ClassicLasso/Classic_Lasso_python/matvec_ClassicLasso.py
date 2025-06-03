def matvec_classic_lasso(y, par, AP):
    """
    Computes Ay = y + sigma * (A * (A^T * y)), matching the MATLAB:
        tmp = AP' * y;
        Ay = y + par.sigma * (AP * tmp);
    
    Parameters
    ----------
    y : np.ndarray, shape (m,)
        Input vector.
    par : dict
        Dictionary containing at least:
            par['sigma'] : float
    AP : np.ndarray, shape (m, n)
        Matrix A.

    Returns
    -------
    Ay : np.ndarray, shape (m,)
        The result y + sigma * (A * (A^T * y)).
    """
    tmp = AP.T.dot(y)                         # A^T * y
    Ay = y + par['sigma'] * (AP.dot(tmp))    # y + sigma * A * tmp
    return Ay
