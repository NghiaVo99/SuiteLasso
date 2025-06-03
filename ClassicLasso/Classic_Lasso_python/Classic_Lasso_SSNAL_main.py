import numpy as np
import time
from Classic_Lasso_SSNCG import *

def proj_inf(v, lam):
    """
    Projection onto the ℓ∞ ball of radius lam:
    Clip each entry of v to lie within [-lam, lam].
    """
    return np.clip(v, -lam, lam)

def Classic_Lasso_SSNAL_main(Amap0, ATmap0, b, lam, parmain, y, xi, x):
    """
    Python translation of the MATLAB function:
    [obj,y,xi,x,info,runhist] = Classic_Lasso_SSNAL_main(Amap0,ATmap0,b,lambda,parmain,y,xi,x)

    Assumes:
      - Amap0, ATmap0 are callables: Amap0(v) ≡ A * v  (or the user‐provided mapping)
        and ATmap0(v) ≡ Aᵀ * v.
      - parmain is a dict containing all fields used below (e.g., 'dscale', 'Lip', etc.).
      - Classic_Lasso_SSNCG, mexscale, mexsigma_update_Classic_Lasso_SSNAL are provided elsewhere in Python,
        with interfaces matching their MATLAB counterparts.
    """
    # Extract parameters from `parmain`
    dscale      = parmain['dscale']        # numpy array or scalar
    Ascaleyes   = parmain['Ascaleyes']     # bool
    m           = parmain['m']
    n           = parmain['n']
    tstart      = parmain['tstart']        # a float timestamp (e.g., time.time())
    existA      = parmain['existA']        # bool: whether a dense matrix A is also provided
    Lip         = parmain['Lip']
    scale       = parmain['scale']
    maxiter     = parmain['maxiter']
    printyes    = parmain['printyes']
    rescale     = parmain['rescale']
    stoptol     = parmain['stoptol']
    orgobjconst = parmain['orgojbconst']

    # If parmain contains a dense matrix 'A', extract it
    if 'A' in parmain:
        A = parmain['A']

    # Compute sigmaLip and adjusted lambda (ld)
    sigmaLip = 1.0 / Lip
    eps = np.finfo(float).eps
    # If dscale ≈ 1, keep ld = lam; otherwise ld = lam * dscale (elementwise)
    if np.linalg.norm(dscale - 1.0) < eps:
        ld = lam
    else:
        ld = lam * dscale
    lambdaorg = lam

    # Keep original b for objective
    borg     = b.copy()
    normborg = 1.0 + np.linalg.norm(borg)

    # Initial A·xi and A·x
    Atxi = ATmap0(xi)
    Ax   = Amap0(x)

    # obj[0] = primal objective; obj[1] = dual objective
    obj = np.zeros(2, dtype=float)
    obj[0] = 0.5 * np.linalg.norm(Ax - borg)**2 + lambdaorg * np.linalg.norm(x, 1) + orgobjconst
    obj[1] = -(0.5 * np.linalg.norm(xi)**2 + borg @ xi) + orgobjconst

    # Initialize scaling factors
    bscale = 1.0
    cscale = 1.0

    # If scale == 1, rescale all quantities accordingly
    if scale == 1:
        factor = np.sqrt(bscale / cscale)
        b    = b / np.sqrt(bscale * cscale)
        xi   = xi / np.sqrt(bscale * cscale)
        Amap = lambda v, Amap0=Amap0, factor=factor: Amap0(v * factor)
        ATmap= lambda v, ATmap0=ATmap0, factor=factor: ATmap0(v * factor)
        if existA:
            A = A * factor
        lam   = lam / cscale
        ld    = ld / cscale
        x     = x / bscale
        y     = y / cscale
        Ax    = Ax / np.sqrt(bscale * cscale)
        Atxi  = Atxi / cscale
    else:
        Amap  = Amap0
        ATmap = ATmap0

    # If Ascaleyes, apply additional diagonal scaling by dscale
    if Ascaleyes:
        Amap  = lambda v, Amap=Amap, dscale=dscale: Amap(dscale * v)
        ATmap = lambda v, ATmap=ATmap, dscale=dscale: dscale * ATmap(v)
        Atxi  = dscale * Atxi
        y     = dscale * y

    # Precompute normb for feasibility checks
    normb = 1.0 + np.linalg.norm(b)

    # Bundle mappings into Ainput_nal
    Ainput_nal = {'Amap': Amap, 'ATmap': ATmap}
    if existA:
        Ainput_nal['A'] = A

    # Initial sigma
    sigma = max(1.0 / np.sqrt(Lip), min(1.0, sigmaLip, lambdaorg))
    if Ascaleyes:
        sigma = 3.0
    if 'sigma' in parmain:
        sigma = parmain['sigma']

    # Initial feasibility/residual calculations
    Rp1        = Ax - b                # A x - b
    Rp         = Rp1 + xi              # primal residual = (Ax - b) + xi
    Rd         = Atxi + y              # dual residual
    primfeas   = np.linalg.norm(Rp) / normb
    dualfeas   = np.linalg.norm(Rd) / (1.0 + np.linalg.norm(y))
    primfeasorg= np.sqrt(bscale * cscale) * np.linalg.norm(Rp) / normborg
    if Ascaleyes:
        dualfeasorg = np.linalg.norm(Rd / dscale) * cscale / (1.0 + np.linalg.norm(y / dscale) * cscale)
    else:
        dualfeasorg = np.linalg.norm(Rd) * cscale / (1.0 + np.linalg.norm(y) * cscale)

    maxfeas     = max(primfeas, dualfeas)
    maxfeasorg  = max(primfeasorg, dualfeasorg)
    relgap      = (obj[0] - obj[1]) / (1.0 + obj[0] + obj[1])

    # Initialize run history dict of lists
    runhist = {
        'dualfeasorg': [dualfeasorg],
        'primfeasorg': [primfeasorg],
        'dualfeas':    [],
        'primfeas':    [],
        'maxfeas':     [],
        'maxfeasorg':  [],
        'sigma':       [],
        'rankS':       [],
        'cnt_Amap':    [],
        'cnt_ATmap':   [],
        'cnt_pAATmap': [],
        'cnt_fAATmap': [],
        'primobj':     [],
        'dualobj':     [],
        'time':        [],
        'relgap':      []
    }

    # Optional header print
    if printyes:
        print("\n" + "*" * 55)
        print("      Classic Lasso: SSNAL")
        print("*" * 55)
        print(f" n = {n:3d}, m = {m:3d}")
        print(f" bscale = {bscale:.2e}, cscale = {cscale:.2e}")
        print("-" * 55)
        print("  iter | [pinfeas  dinfeas] [pinforg  dinforg]   relgaporg |    pobj       dobj   | time | sigma | rankS |")
        print("*" * 55)
        t_elapsed = time.time() - tstart
        print(f"   0  | {primfeas:.2e}   {dualfeas:.2e}   {primfeasorg:.2e}   {dualfeasorg:.2e}  {relgap:.2e}  {obj[0]:.7e}  {obj[1]:.7e}  {t_elapsed:5.1f}  {sigma:.2e}")

    # Set up SSNCG‐specific parameters
    SSNCG = True
    if SSNCG:
        parNCG = {
            'sigma':   sigma,
            'tolconst': 0.5,
            'n':       n
        }

    maxitersub = 10
    breakyes   = False
    prim_win   = 0
    dual_win   = 0

    # ssncgop options
    ssncgop = {
        'existA':   existA,
        'tol':      stoptol,
        'precond':  0,
        'bscale':   bscale,
        'cscale':   cscale,
        'printsub': printyes,
        'Ascaleyes': 1 if Ascaleyes else 0
    }
    if Ascaleyes:
        ssncgop['dscale'] = dscale

    sigmamax = 1e7
    sigmamin = 1e-4
    if Ascaleyes:
        sigmamax *= np.mean(dscale)

    # Begin main SSNAL iteration loop
    for it in range(1, maxiter + 1):
        # (1) Optional rescaling step
        # MATLAB condition:
        # if ((rescale == 1) && (maxfeas < 5e2) && (rem(iter,3)==1) && (iter>1)) ...
        #  || (~existA && ((rescale>=2) && maxfeas<1e-1 && abs(relgap)<0.05 ...
        #      && iter>=5 && (max(normx/normyxi, normyxi/normx)>1.7) && rem(iter,5)==1))
        cond1 = (rescale == 1) and (maxfeas < 5e2) and ((it > 1) and (it % 3 == 1))
        # For cond2, we need normx, normAtxi, normyxi from a previous iteration. We'll guard access via existA.
        cond2 = False
        if (not existA) and (rescale >= 2) and (maxfeas < 1e-1) and (abs(relgap) < 0.05) and (it >= 5):
            # We must have computed normx, normAtxi already in a prior iteration:
            try:
                normx = np.linalg.norm(x)
                normAtxi = np.linalg.norm(Atxi)
                normyxi = max(normx, normAtxi)
                ratio = max(normx / normyxi, normyxi / normx) if (normx > 0 and normyxi > 0) else 0.0
                cond2 = (ratio > 1.7) and (it % 5 == 1)
            except NameError:
                cond2 = False

        if cond1 or cond2:
            # Compute norms needed for re‐scaling
            normy = np.linalg.norm(y)
            normAtxi = np.linalg.norm(Atxi)
            normx = np.linalg.norm(x)
            normyxi = max(normx, normAtxi)

            # Call user‐provided `mexscale`; 
            # Expected signature (MATLAB): 
            #   [sigma, bscale2, cscale2, sbc, sboc, bscale, cscale] = mexscale(sigma, normx, normyxi, bscale, cscale);
            sigma, bscale2, cscale2, sbc, sboc, bscale, cscale = mexscale(
                sigma, normx, normyxi, bscale, cscale
            )

            # Update Amap, ATmap by scaling input arguments
            Amap_old  = Amap
            ATmap_old = ATmap
            Amap  = lambda v, Amap_old=Amap_old, sboc=sboc: Amap_old(v * sboc)
            ATmap = lambda v, ATmap_old=ATmap_old, sboc=sboc: ATmap_old(v * sboc)
            Ainput_nal['Amap']  = Amap
            Ainput_nal['ATmap'] = ATmap

            if existA:
                A = A * sboc
                Ainput_nal['A'] = A

            # Rescale vectors and parameters
            b     = b / sbc
            ld    = ld / cscale2
            x     = x / bscale2
            xi    = xi / sbc
            Atxi  = Atxi / cscale2
            Ax    = Ax / sbc

            # Update ssncgop scaling records
            ssncgop['bscale'] = bscale
            ssncgop['cscale'] = cscale

            # Recompute normb
            normb = 1.0 + np.linalg.norm(b)

            if printyes:
                print(f"\n  [rescale={rescale:.0f}: {it:3d} | normx={normx:.2e}, normAtxi={normAtxi:.2e}, normy={np.linalg.norm(y):.2e} | bscale={bscale:.2e}, cscale={cscale:.2e}, sigma={sigma:.2e}]")

            rescale += 1
            prim_win = 0
            dual_win = 0

        # (2) Save previous x
        xold = x.copy()

        # (3) Adjust maxitersub based on current dual feasibility
        if dualfeas < 1e-5:
            maxitersub = max(maxitersub, 30)
        elif dualfeas < 1e-3:
            maxitersub = max(maxitersub, 30)
        elif dualfeas < 1e-1:
            maxitersub = max(maxitersub, 20)

        ssncgop['maxitersub'] = maxitersub
        parNCG['sigma']       = sigma

        # (4) Call the SSNCG subroutine (user must supply this function in Python)
        #     Expected signature in Python:
        #       y, Atxi, xi, parNCG, runhist_NCG, info_NCG = Classic_Lasso_SSNCG(
        #           n, b, Ainput_nal, x, Ax, Atxi, xi, ld, parNCG, ssncgop
        #       )
        y, Atxi, xi, parNCG, runhist_NCG, info_NCG = classic_lasso_ssncg(
            n, b, Ainput_nal, x, Ax, Atxi, xi, ld, parNCG, ssncgop
        )

        # (5) If the sub‐solver flagged breakyes < 0, tighten tolerance constant
        if info_NCG['breakyes'] < 0:
            parNCG['tolconst'] = max(parNCG['tolconst'] / 1.06, 1e-3)

        # (6) Update dual residual and x, Ax
        Rd = Atxi + y
        x  = -sigma * info_NCG['ytmp']
        Ax = info_NCG['Ax']

        normRd   = np.linalg.norm(Rd)
        normy    = np.linalg.norm(y)
        dualfeas = normRd / (1.0 + normy)

        if Ascaleyes:
            dualfeasorg = np.linalg.norm(Rd / dscale) * cscale / (1.0 + np.linalg.norm(y / dscale) * cscale)
        else:
            dualfeasorg = normRd * cscale / (1.0 + normy * cscale)

        # (7) Update primal residual, feasibilities, gaps
        Rp1       = Ax - b
        Rp        = Rp1 + xi
        normRp    = np.linalg.norm(Rp)
        primfeas  = normRp / normb
        primfeasorg = np.sqrt(bscale * cscale) * normRp / normborg
        maxfeas     = max(primfeas, dualfeas)
        maxfeasorg  = max(primfeasorg, dualfeasorg)

        # (8) Record run history for this iteration
        runhist['dualfeas'].append(dualfeas)
        runhist['primfeas'].append(primfeas)
        runhist['maxfeas'].append(maxfeas)
        runhist['primfeasorg'].append(primfeasorg)
        runhist['dualfeasorg'].append(dualfeasorg)
        runhist['maxfeasorg'].append(maxfeasorg)
        runhist['sigma'].append(sigma)
        # parNCG['rr'] is assumed to be a numpy array or list where "sum(1 - parNCG['rr'])" is valid
        runhist['rankS'].append(int(np.sum(1.0 - parNCG['rr'])))
        runhist['cnt_Amap'].append(info_NCG['cnt_Amap'])
        runhist['cnt_ATmap'].append(info_NCG['cnt_ATmap'])
        runhist['cnt_pAATmap'].append(info_NCG['cnt_pAATmap'])
        runhist['cnt_fAATmap'].append(info_NCG['cnt_fAATmap'])

        # (9) Check for termination (primal+dual feasibility and optimality condition)
        eta = None
        if max(primfeasorg, dualfeasorg) < 500.0 * max(1e-6, stoptol):
            grad = ATmap0(Rp1 * np.sqrt(bscale * cscale))
            if Ascaleyes:
                etaorg = np.linalg.norm(grad + proj_inf(dscale * x * bscale - grad, lambdaorg))
                eta    = etaorg / (1.0 + np.linalg.norm(grad) + np.linalg.norm(dscale * x * bscale))
            else:
                etaorg = np.linalg.norm(grad + proj_inf(x * bscale - grad, lambdaorg))
                eta    = etaorg / (1.0 + np.linalg.norm(grad) + np.linalg.norm(x * bscale))

            if eta < stoptol:
                breakyes = True
                msg = 'converged'

        # (10) Print iteration summary if requested
        if printyes:
            objscale = bscale * cscale
            primobj  = objscale * (0.5 * np.linalg.norm(Rp1)**2 + np.linalg.norm(ld * x, 1)) + orgobjconst
            dualobj  = objscale * (-0.5 * np.linalg.norm(xi)**2 + borg @ xi) + orgobjconst
            relgap   = (primobj - dualobj) / (1.0 + abs(primobj) + abs(dualobj))
            ttime    = time.time() - tstart

            print(f"\n {it:5d} | [{primfeas:.2e} {dualfeas:.2e}]    [{primfeasorg:.2e} {dualfeasorg:.2e}]   {relgap:.2e} | {primobj:.4e}  {dualobj:.4e} | {ttime:5.1f} | {sigma:.2e} | sigamorg = {sigma * bscale/cscale:.2e} | ", end='')

            if it >= 1:
                print(f"{int(np.sum(1.0 - parNCG['rr']))} |", end='')

            if eta is not None:
                print(f"\n       [eta = {eta:.2e}, etaorg = {etaorg:.2e}]", end='')

            if it % 3 == 1:
                normx     = np.linalg.norm(x)
                normAtxi  = np.linalg.norm(Atxi)
                normy_now = np.linalg.norm(y)
                print(f"\n       [normx, Atxi, y = {normx:.2e} {normAtxi:.2e} {normy_now:.2e}]", end='')

            # Record objectives and time into run history
            runhist['primobj'].append(primobj)
            runhist['dualobj'].append(dualobj)
            runhist['time'].append(ttime)
            runhist['relgap'].append(relgap)

        # (11) Break if converged
        if breakyes:
            if printyes:
                print(f"\n  breakyes = {1.0}, {msg}")
            break

        # (12) Update winning counts and sigma via `mexsigma_update_Classic_Lasso_SSNAL`
        if primfeasorg < dualfeasorg:
            prim_win += 1
        else:
            dual_win += 1

        sigma, prim_win, dual_win = mexsigma_update_Classic_Lasso_SSNAL(
            sigma, sigmamax, sigmamin, prim_win, dual_win, it, info_NCG['breakyes']
        )

    # (13) Post‐loop housekeeping
    if not printyes:
        ttime = time.time() - tstart

    if (it == maxiter) and (not breakyes):
        msg = 'maximum number of iterations reached'

    # Build the info dictionary to return
    info = {
        'iter': it,
        'bscale': bscale,
        'cscale': cscale,
        'Ax': Ax,
        'ttime': ttime,
        'msg': msg
    }

    return obj, y, xi, x, info, runhist
