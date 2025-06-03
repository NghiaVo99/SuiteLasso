def matvec_ip_xi(y, par, Ainput):
    """
    Computes Ay = y + sigma * A * ((1 - rr) .* (A^T * y)), matching the MATLAB:
        tmp = Ainput.ATmap(y);
        tmp = (1 - par.rr) .* tmp;
        Ay = y + par.sigma * Ainput.Amap(tmp);

    Parameters
    ----------
    y : np.ndarray, shape (m,)
        Input vector.
    par : dict
        Dictionary containing:
            par['sigma'] : float
            par['rr']    : np.ndarray, same length as A^T*y
    Ainput : object
        Object providing two callables:
            Ainput.ATmap(x) : returns A^T @ x
            Ainput.Amap(x)  : returns A @ x

    Returns
    -------
    Ay : np.ndarray, shape (m,)
        The result y + sigma * A * ((1 - rr) * (A^T * y)).
    """
    tmp = Ainput.ATmap(y)               # Compute A^T * y
    tmp = (1.0 - par['rr']) * tmp      # Elementwise multiply by (1 - rr)
    Ay = y + par['sigma'] * Ainput.Amap(tmp)
    return Ay
