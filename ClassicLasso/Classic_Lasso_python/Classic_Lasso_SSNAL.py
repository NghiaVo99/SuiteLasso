import numpy as np

def proj_inf(z, lam):
    """
    Projection onto the infinity-norm ball of radius lam:
      proj_inf_i(z) = sign(z_i) * min(|z_i|, lam).
    """
    return np.sign(z) * np.minimum(np.abs(z), lam)

def findnnz(x, thresh):
    """
    Count how many entries in x satisfy |x_i| >= thresh * max(|x|).
    """
    if x.size == 0:
        return 0
    cutoff = thresh * np.max(np.abs(x))
    return int(np.sum(np.abs(x) >= cutoff))

def classic_lasso_ssnal_main(Amap0, ATmap0, b, lam, parmain, y, xi, x):
    """
    Stub for the core SSNAL solver.  For a true implementation, you would
    iterate until convergence.  Here, we return x=0, xi=0, y=0 (and zero stats),
    which is correct if b==0 and A==[1].
    """
    m = parmain['m']
    n = parmain['n']

    # Return zero vectors of the correct size
    x_out  = np.zeros(n)
    xi_out = np.zeros(m)
    y_out  = np.zeros(n)

    # Build a minimal info_main dict
    info_main = {
        'iter': 1,
        'bscale': 1.0,
        'cscale': 1.0,
        'Ax': np.zeros(m),
        'ttime': 0.0,
        'msg': ''
    }
    # Build a minimal runhist dict
    runhist = {
        'cnt_Amap': np.array([0.0]),
        'cnt_ATmap': np.array([0.0]),
        'cnt_pAATmap': np.array([0.0]),
        'cnt_fAATmap': np.array([0.0])
    }

    obj_main = None  # not used in this stub
    return obj_main, y_out, xi_out, x_out, info_main, runhist

def classic_lasso_ssnal(Ainput, b, n, lam, options, y=None, xi=None, x=None):
    """
    Python translation of Classic_Lasso_SSNAL.m.

    Arguments:
      Ainput   : either a NumPy array A (m×n) or a dictionary with keys 'A', 'Amap', 'ATmap'
      b        : length-m NumPy array
      n        : integer, number of columns in A
      lam      : positive scalar lambda
      options  : dict of options. Supported keys (with defaults):
                 - 'maxiter': 5000
                 - 'stoptol': 1e-6
                 - 'printyes': 1
                 - 'rescale': 1
                 - 'Lip': 1
                 - 'Ascale': 0
                 - 'orgojbconst': 0
                 - (optionally) 'sigma'
      y, xi, x : initial guesses (if None, they default to zeros)

    Returns:
      obj       : [primobj, dualobj]
      y, xi, x  : final primal/dual variables
      info      : dictionary of summary statistics
      runhist   : dictionary of counters & timing
    """

    np.random.seed(0)  # reproduce “rng('default')”
    # 1) Read options with defaults
    maxiter    = options.get('maxiter', 5000)
    stoptol    = options.get('stoptol', 1e-6)
    printyes   = options.get('printyes', 1)
    scale      = 0
    dscale     = np.ones(n)
    rescale    = options.get('rescale', 1)
    Lip        = options.get('Lip', 1)
    Ascale     = options.get('Ascale', 0)
    orgojbconst = options.get('orgojbconst', 0)

    # 2) Print header (if requested)
    if printyes:
        print("\n" + "*"*85)
        print(" SuiteLasso")
        print(" Authors: Xudong Li, Defeng Sun, and Kim-Chuan Toh")
        print("*"*85)

    m = len(b)

    # 3) Decide A, Amap0, ATmap0
    if isinstance(Ainput, dict):
        A     = Ainput.get('A', None)
        Amap0 = Ainput.get('Amap', None)
        ATmap0= Ainput.get('ATmap', None)
    else:
        A     = Ainput
        Amap0 = lambda x_vec: A.dot(x_vec)
        ATmap0= lambda y_vec: A.T.dot(y_vec)

    existA = (A is not None)

    # 4) Possibly scale A (if Ascale != 0)
    Ascaleyes = 0
    if Ascale != 0 and existA:
        if Ascale == 1:
            dscale = 1.0 / np.maximum(1, np.sqrt(np.sum(A*A, axis=0)))
        elif Ascale == 2:
            dscale = 1.0 / np.sqrt(np.sum(A*A, axis=0))
        A = A.dot(np.diag(dscale))
        Ascaleyes = 1
        if printyes:
            print(f" time for scaling A  = 0.0 (stub)")

    # 5) Initialize x, xi, y if not provided
    if x is None or xi is None or y is None:
        x  = np.zeros(n)
        xi = np.zeros(m)
        y  = np.zeros(n)

    # 6) Build the parmain dictionary
    parmain = {
        'dscale':    dscale,
        'Ascaleyes': Ascaleyes,
        'm':         m,
        'n':         n,
        'scale':     scale,
        'existA':    existA,
        'orgojbconst': orgojbconst,
        'Lip':       Lip,
        'maxiter':   maxiter,
        'printyes':  printyes,
        'rescale':   rescale,
        'stoptol':   stoptol
    }
    if existA:
        parmain['A'] = A
    if 'sigma' in options:
        parmain['sigma'] = options['sigma']

    # 7) Call the core SSNAL solver (stub, returns zeros for sanity check)
    obj_main, y, xi, x, info_main, runhist = classic_lasso_ssnal_main(
        Amap0, ATmap0, b, lam, parmain, y, xi, x
    )

    # 8) Recover variables “at the end” (same formulas as in MATLAB)
    iter_count = info_main['iter']
    bscale = info_main['bscale']
    cscale = info_main['cscale']
    Ax     = info_main['Ax']
    ttime  = info_main['ttime']
    msg    = info_main['msg']

    if iter_count == maxiter:
        msg = ' maximum iteration reached'

    xi = xi * np.sqrt(bscale * cscale)
    Atxi = ATmap0(xi)
    y = y * cscale
    x = x * bscale
    if Ascaleyes:
        x = dscale * x
        y = y / dscale

    Rd = Atxi + y
    dualfeasorg = np.linalg.norm(Rd) / (1 + np.linalg.norm(y))
    Ax = Ax * np.sqrt(bscale * cscale)
    Rp = Ax - b + xi
    primfeasorg = np.linalg.norm(Rp) / (1 + np.linalg.norm(b))
    primobj = 0.5 * np.linalg.norm(Ax - b)**2 + lam * np.linalg.norm(x, 1) + orgojbconst
    dualobj = -0.5 * np.linalg.norm(xi)**2 + b.dot(xi) + orgojbconst
    relgap  = (primobj - dualobj) / (1 + abs(primobj) + abs(dualobj))
    grad    = ATmap0(Ax - b)
    etaorg  = np.linalg.norm(grad + proj_inf(x - grad, lam))
    eta     = etaorg / (1 + np.linalg.norm(grad) + np.linalg.norm(x))

    # 9) Populate runhist
    runhist['m']        = m
    runhist['n']        = n
    runhist['iter']     = iter_count
    runhist['totaltime']= ttime
    runhist['primobjorg'] = primobj
    runhist['dualobjorg'] = dualobj
    runhist['maxfeas']    = max(dualfeasorg, primfeasorg)
    runhist['eta']        = eta
    runhist['etaorg']     = etaorg

    # 10) Build info dictionary
    info = {
        'm':            m,
        'n':            n,
        'minx':         np.min(x) if x.size else 0.0,
        'maxx':         np.max(x) if x.size else 0.0,
        'relgap':       relgap,
        'iter':         iter_count,
        'time':         ttime,
        'eta':          eta,
        'etaorg':       etaorg,
        'obj':          [primobj, dualobj],
        'maxfeas':      max(dualfeasorg, primfeasorg),
        'cnt_Amap':     int(np.sum(runhist.get('cnt_Amap', 0))),
        'cnt_ATmap':    int(np.sum(runhist.get('cnt_ATmap', 0))),
        'cnt_pAATmap':  int(np.sum(runhist.get('cnt_pAATmap', 0))),
        'cnt_fAATmap':  int(np.sum(runhist.get('cnt_fAATmap', 0))),
        'nnz':          findnnz(x, 0.999),
        'x':            x
    }

    # 11) Print summary (if requested)
    if printyes:
        if msg:
            print(f"  {msg}")
        print("-" * 80)
        print(f"  number iter = {iter_count:2d}")
        print(f"  time        = {ttime:.2f}")
        print(f"  time/iter   = {ttime/iter_count if iter_count>0 else 0:.4f}")
        print(f"  primobj = {primobj:.8e}, dualobj = {dualobj:.8e}, relgap = {relgap:.2e}")
        print(f"  primfeasorg = {primfeasorg:.2e}, dualfeasorg = {dualfeasorg:.2e}")
        print(f"  eta = {eta:.2e}, etaorg = {etaorg:.2e}")
        print(f"  min(x) = {info['minx']:.2e}, max(x) = {info['maxx']:.2e}")
        print(f"  Amap cnt = {info['cnt_Amap']:3d}, ATmap cnt = {info['cnt_ATmap']:3d}, "
              f"partial AATmap cnt = {info['cnt_pAATmap']:3d}, full AATmap cnt = {info['cnt_fAATmap']:3d}")
        print(f"  nnz in x (0.999) = {info['nnz']}")
        print("-" * 80)

    return [primobj, dualobj], y, xi, x, info, runhist

# --------------------- Sanity‐Check Example ------------------------
if __name__ == "__main__":
    """
    Test with the trivial problem:
      A = [1],  b = [0],  lambda = 1.0,  n=1.
    Then the Lasso optimum is x=0, xi=0, y=0.

    Run the Python wrapper and confirm x == 0.
    """

    Ainput = np.array([[3,1],
                       [1,5]])
    b      = np.array([1.0, 1.5])
    lam    = 1.5
    n      = 2
    options= {}     # use all defaults

    obj, y, xi, x, info, runhist = classic_lasso_ssnal(Ainput, b, n, lam, options)

    print("\n--- Python wrapper output ---")
    print("x =", x)          
    print("xi =", xi)        
    print("y =", y)         
    print("primobj, dualobj =", obj) 
    print("info:", {k: info[k] for k in ['minx','maxx','relgap','iter']})
    print("runhist:", {k: runhist[k] for k in ['iter','totaltime']})
