import numpy as np

def classic_lasso_ssncg(n, b, Ainput, x0, Ax0, Atxi0, xi0, ld, par, options):
    """
    A Python translation of the MATLAB function Classic_Lasso_SSNCG.

    Solves the subproblem in the SSNAL (semismooth Newton augmented Lagrangian)
    method for Lasso:
        max_{xi, y}  { b^T xi - (1/2)||xi||^2 - (σ/2)||y - ( -A^T xi - x0/σ )||^2 }
        subject to  ||y||_∞ <= ld
    and updates (y, Atxi, xi, par, runhist, info).

    Parameters
    ----------
    n : int
        Number of features (dimension of x and y).
    b : np.ndarray, shape (m,)
        The observed data vector.
    Ainput : object
        Object providing two callables:
            Ainput.Amap(x): computes A @ x (shape (m,))
            Ainput.ATmap(x): computes A^T @ x (shape (n,))
    x0 : np.ndarray, shape (n,)
        Current primal iterate.
    Ax0 : np.ndarray, shape (m,)
        A @ x0.
    Atxi0 : np.ndarray, shape (n,)
        A^T @ xi0.
    xi0 : np.ndarray, shape (m,)
        Current dual variable xi.
    ld : float
        The L∞-norm bound for y.
    par : dict
        Dictionary containing algorithmic parameters; must include:
            par['sigma']    : σ parameter in SSNAL
            par['tolconst'] : multiplier for the subproblem tolerance
            (other fields may be set/updated)
    options : dict
        Dictionary of options; may include:
            'printsub'   : bool
            'maxitersub' : int
            'tiny'       : float
            'tol'        : float
            'precond'    : int (0 or 1)
            'Ascaleyes'  : int (0 or 1)
            'existA'     : bool
            'bscale'     : float
            'cscale'     : float
            'dscale'     : np.ndarray (used if Ascaleyes == 1)
    Returns
    -------
    y : np.ndarray, shape (n,)
        Updated dual variable y.
    Atxi : np.ndarray, shape (n,)
        A^T @ xi at the final iterate.
    xi : np.ndarray, shape (m,)
        Updated dual variable xi.
    par : dict
        Updated `par` dictionary (with updated fields like 'rr', 'iter', etc.).
    runhist : dict
        History of the subproblem iterations, containing lists:
            'psqmr'    : number of PSQMR iterations each sub-iteration
            'findstep' : number of line-search steps each sub-iteration
            'priminf'  : primal infeasibility measure
            'dualinf'  : dual infeasibility measure
            'Ly'       : objective value L(·) at each sub-iteration
            'solve_ok' : status code from linear solve (per iteration)
    info : dict
        Information about the subproblem solve, including:
            'maxCG', 'avgCG', 'breakyes', 'itersub', 'tolconst',
            'RpGradratio', 'rankX', 'ytmp', 'cnt_Amap', 'cnt_ATmap',
            'Ax', 'cnt_pAATmap', 'cnt_fAATmap'.
    """

    # --- 1. Default parameters and option extraction -----------------------
    printsub     = options.get('printsub', True)
    maxitersub   = options.get('maxitersub', 50)
    tiny         = options.get('tiny', 1e-10)
    tol          = options.get('tol', 1e-6)
    maxitpsqmr   = options.get('maxitpsqmr', 500)
    precond      = options.get('precond', 0)
    Ascaleyes    = options.get('Ascaleyes', 0)
    existA       = options.get('existA', False)
    bscale       = options.get('bscale', 1.0)
    cscale       = options.get('cscale', 1.0)

    sig          = par['sigma']
    normborg     = 1.0 + np.linalg.norm(b) * np.sqrt(bscale * cscale)

    # Function handles (callables) for A and A^T
    Amap  = lambda x: Ainput.Amap(x)    # returns shape (m,)
    ATmap = lambda x: Ainput.ATmap(x)   # returns shape (n,)

    # Initial projection onto the infinity-norm ball: y = proj_inf( yinput, ld )
    # where yinput = -Atxi0 - x0/sig
    yinput = -Atxi0 - x0 / sig
    y, rr  = proj_inf(yinput, ld)       # proj_inf should return (y_proj, rr_indicator)
    par['rr'] = rr

    # Residual in xi: R_p = A x0 - b + xi0
    Rpb      = Ax0 - b + xi0
    normRp   = np.linalg.norm(Rpb)

    # Temporary vector ytmp = yinput - y
    ytmp     = yinput - y

    # Initialize Atxi and xi
    Atxi     = Atxi0.copy()
    xi       = xi0.copy()

    # Initial augmented Lagrangian objective L(y, xi)
    Ly       = b.dot(xi) - 0.5 * np.linalg.norm(xi)**2 - 0.5 * sig * np.linalg.norm(ytmp)**2

    # Initialize run history
    runhist = {
        'psqmr':    np.zeros(maxitersub, dtype=int),
        'findstep': np.zeros(maxitersub, dtype=int),
        'priminf':  np.zeros(maxitersub),
        'dualinf':  np.zeros(maxitersub),
        'Ly':       np.zeros(maxitersub),
        'solve_ok': np.zeros(maxitersub, dtype=int)
    }

    # Counters for operator calls
    cnt_Amap     = 0
    cnt_ATmap    = 0
    cnt_pAATmap  = 0
    cnt_fAATmap  = 0

    breakyes = 0

    # --- 2. Main Newton iteration (subproblem loop) ------------------------
    for itersub in range(1, maxitersub + 1):
        yold    = y.copy()
        xiold   = xi.copy()
        Atxiold = Atxi.copy()

        # Dual residual: R_d = A^T xi + y
        Rdz    = Atxi + y
        normRd = np.linalg.norm(Rdz)

        # Compute gradient of L w.r.t. xi: GradLxi = -(xi - b + (-σ A ytmp))
        msigAytmp   = -sig * Amap(ytmp)
        cnt_Amap   += 1
        GradLxi     = -(xi - b + msigAytmp)
        normGradLxi = np.linalg.norm(GradLxi) * np.sqrt(bscale * cscale) / normborg

        priminf_sub = normGradLxi
        if Ascaleyes:
            dualinf_sub = np.linalg.norm(Rdz / options['dscale']) * cscale / (
                          1.0 + np.linalg.norm(y / options['dscale']) * cscale)
        else:
            dualinf_sub = normRd * cscale / (1.0 + np.linalg.norm(y) * cscale)

        # Determine subproblem tolerance: tolsub = max(min(1, par['tolconst'] * dualinf_sub), tolsubconst * tol)
        if max(priminf_sub, dualinf_sub) < tol:
            tolsubconst = 0.9
        else:
            tolsubconst = 0.05

        tolsub = max(min(1.0, par['tolconst'] * dualinf_sub), tolsubconst * tol)

        # Record history
        runhist['priminf'][itersub - 1] = priminf_sub
        runhist['dualinf'][itersub - 1] = dualinf_sub
        runhist['Ly'][itersub - 1]      = Ly

        # Print iteration info if requested
        if printsub:
            print(f"\n  {itersub:2d}  {Ly:12.10e}  {priminf_sub:.2e}  {dualinf_sub:.2e}  {par['tolconst']:.2e}", end='')

        # Check for termination of the subproblem
        if (normGradLxi < tolsub) and (itersub > 1):
            msg = "good termination in subproblem:"
            if printsub:
                print(f"\n   {msg}  dualinfes = {dualinf_sub:.2e}, gradLyxi = {normGradLxi:.2e}, tolsub = {tolsub:.2e}", end='')
            breakyes = -1
            itersub_final = itersub
            break

        # --- 3. Compute Newton direction via PSQMR solver --------------------
        # Update par for linear solver
        par['epsilon'] = min(1e-3, 0.1 * normGradLxi)
        par['precond'] = precond

        # Adapt maxitpsqmr based on dualinf_sub and iteration count
        if (dualinf_sub > 1e-3) or (itersub <= 5):
            maxitpsqmr = max(maxitpsqmr, 200)
        elif dualinf_sub > 1e-4:
            maxitpsqmr = max(maxitpsqmr, 300)
        elif dualinf_sub > 1e-5:
            maxitpsqmr = max(maxitpsqmr, 400)
        elif dualinf_sub > 5e-6:
            maxitpsqmr = max(maxitpsqmr, 500)

        if itersub > 1:
            prim_ratio = priminf_sub / runhist['priminf'][itersub - 2]
            dual_ratio = dualinf_sub / runhist['dualinf'][itersub - 2]
        else:
            prim_ratio = 0.0
            dual_ratio = 0.0

        rhs = GradLxi.copy()
        tolpsqmr = min(5e-3, 0.1 * np.linalg.norm(rhs))
        const2   = 1.0

        if (itersub > 1) and ((prim_ratio > 0.5) or (priminf_sub > 0.1 * runhist['priminf'][0])):
            const2 *= 0.5
        if dual_ratio > 1.1:
            const2 *= 0.5

        tolpsqmr *= const2
        par['tol']   = tolpsqmr
        par['maxit'] = maxitpsqmr

        # Solve for dxi:     H * dxi = GradLxi
        # using a separate function Classic_Lasso_linsys_solver
        dxi, resnrm, solve_ok = classic_lasso_linsys_solver(Ainput, rhs, par)
        # dxi: np.ndarray shape (m,)
        # resnrm: list or array with residual norms from PSQMR (length = iterpsqmr + 1)
        # solve_ok: integer flag

        Atdxi = ATmap(dxi)
        cnt_ATmap += 1

        iterpsqmr = len(resnrm) - 1
        if iterpsqmr == 0:
            cnt_pAATmap += 1
        else:
            if existA:
                cnt_pAATmap += iterpsqmr
            else:
                cnt_fAATmap += iterpsqmr

        if printsub:
            print(f" | {par['tol']:.1e} {resnrm[-1]:.1e} {iterpsqmr:3d} {const2:.1f} {int(np.sum(1 - par['rr'])):2d}", end='')

        par['iter'] = itersub

        # Choose line-search type
        if (itersub <= 3 and dualinf_sub > 1e-4) or (par['iter'] < 3):
            stepop = 1
        else:
            stepop = 2

        step_op = {'stepop': stepop}
        steptol = 1e-5

        # --- 4. Line search to update (xi, y, Atxi, Ly) ----------------------
        (par, Ly, xi, Atxi, y, ytmp, alp, iterstep) = findstep(
            par, b, ld, Ly, xi, Atxi, y, ytmp, dxi, Atdxi, steptol, step_op
        )

        runhist['solve_ok'][itersub - 1] = solve_ok
        runhist['psqmr'][itersub - 1]    = iterpsqmr
        runhist['findstep'][itersub - 1] = iterstep

        if alp < tiny:
            breakyes = 11
            itersub_final = itersub
            break

        if printsub:
            print(f"  {alp:.2e} {iterstep:2d}", end='')

        # --- 5. Check for stagnation and other exit conditions ---------------
        if itersub > 4:
            idx = list(range(max(1, itersub - 3) - 1, itersub))  # zero-based indices
            tmp = runhist['priminf'][idx]
            ratio = np.min(tmp) / np.max(tmp)
            solve_ok_idx = runhist['solve_ok'][idx]
            if (np.all(solve_ok_idx <= -1) and (ratio > 0.9) and
                (np.min(runhist['psqmr'][idx]) == np.max(runhist['psqmr'][idx])) and
                (np.max(tmp) < 5 * tol)):
                if printsub:
                    print("#", end='')
                breakyes = 1
                itersub_final = itersub
                break

            const3 = 0.7
            half_point = int(np.ceil(itersub * const3))
            priminf_1half = np.min(runhist['priminf'][:half_point])
            priminf_2half = np.min(runhist['priminf'][half_point:itersub])
            priminf_best  = np.min(runhist['priminf'][:itersub - 1])
            priminf_ratio = runhist['priminf'][itersub - 1] / runhist['priminf'][itersub - 2]
            dualinf_ratio = runhist['dualinf'][itersub - 1] / runhist['dualinf'][itersub - 2]
            stagnate_idx  = np.where(runhist['solve_ok'][:itersub] <= -1)[0]
            stagnate_count = len(stagnate_idx)
            idx2 = list(range(max(1, itersub - 7) - 1, itersub))

            if (itersub >= 10 and np.all(runhist['solve_ok'][idx2] == -1) and
                (priminf_best < 1e-2) and (dualinf_sub < 1e-3)):
                tmp2 = runhist['priminf'][idx2]
                ratio2 = np.min(tmp2) / np.max(tmp2)
                if ratio2 > 0.5:
                    if printsub:
                        print("##", end='')
                    breakyes = 2
                    itersub_final = itersub
                    break

            if (itersub >= 15 and (priminf_1half < min(2e-3, priminf_2half)) and
                (dualinf_sub < 0.8 * runhist['dualinf'][0]) and (dualinf_sub < 1e-3) and
                (stagnate_count >= 3)):
                if printsub:
                    print("###", end='')
                breakyes = 3
                itersub_final = itersub
                break

            if (itersub >= 15 and (priminf_ratio < 0.1) and
                (priminf_sub < 0.8 * priminf_1half) and
                (dualinf_sub < min(1e-3, 2 * priminf_sub)) and
                ((priminf_sub < 2e-3) or (dualinf_sub < 1e-5 and priminf_sub < 5e-3)) and
                (stagnate_count >= 3)):
                if printsub:
                    print(" $$", end='')
                breakyes = 4
                itersub_final = itersub
                break

            if (itersub >= 10 and (dualinf_sub > 5 * np.min(runhist['dualinf'][:itersub])) and
                (priminf_sub > 2 * np.min(runhist['priminf'][:itersub]))):
                if printsub:
                    print("$$$", end='')
                breakyes = 5
                itersub_final = itersub
                break

            if itersub >= 20:
                dual_ratio_all = runhist['dualinf'][1:itersub] / runhist['dualinf'][:itersub - 1]
                idx_incr = np.where(dual_ratio_all > 1)[0]
                if len(idx_incr) >= 3:
                    dualinf_increment = np.mean(dual_ratio_all[idx_incr])
                    if dualinf_increment > 1.25:
                        if printsub:
                            print("^^", end='')
                        breakyes = 6
                        itersub_final = itersub
                        break

            if breakyes > 0:
                Rdz    = Atxi + y
                msigAytmp = -sig * Amap(ytmp)
                cnt_Amap += 1
                if printsub:
                    if Ascaleyes:
                        dualfeasorg = np.linalg.norm(Rdz / options['dscale']) * cscale / (
                                      1.0 + np.linalg.norm(y / options['dscale']) * cscale)
                    else:
                        dualfeasorg = np.linalg.norm(Rdz) * cscale / (1.0 + np.linalg.norm(y) * cscale)
                    print(f"\n new dualfeasorg = {dualfeasorg:.2e}", end='')
                break

        else:
            # If the for-loop completes without break, set final iteration count
            itersub_final = maxitersub

    # --- 6. After exiting the loop, collect info -----------------------------
    if breakyes == 0:
        itersub_final = maxitersub

    info = {
        'maxCG':       int(np.max(runhist['psqmr'][:itersub_final])),
        'avgCG':       float(np.sum(runhist['psqmr'][:itersub_final]) / itersub_final),
        'breakyes':    breakyes,
        'itersub':     itersub_final,
        'tolconst':    par['tolconst'],
        'RpGradratio': normRp * np.sqrt(bscale * cscale) / (normGradLxi * normborg),
        'rankX':       par['rr'],
        'ytmp':        ytmp,
        'cnt_Amap':    cnt_Amap,
        'cnt_ATmap':   cnt_ATmap,
        'Ax':          msigAytmp,
        'cnt_pAATmap': cnt_pAATmap,
        'cnt_fAATmap': cnt_fAATmap
    }

    return y, Atxi, xi, par, runhist, info


def findstep(par, b, ld, Ly0, xi0, Atxi0, y0, ytmp0, dxi, Atdxi, tol, options):
    """
    Python translation of the nested MATLAB function findstep.
    Performs a backtracking/bi-section line search to find step length alpha
    along direction dxi, updating (xi, Atxi, y, ytmp, Ly).

    Parameters
    ----------
    par : dict
        Algorithmic parameters; must contain 'sigma' (σ).
        Will be updated with 'rr' and other fields.
    b : np.ndarray, shape (m,)
        Observed data vector.
    ld : float
        L∞-norm bound for y.
    Ly0 : float
        Objective value L at the current iterate (before update).
    xi0 : np.ndarray, shape (m,)
        Current dual variable ξ.
    Atxi0 : np.ndarray, shape (n,)
        A^T @ xi0.
    y0 : np.ndarray, shape (n,)
        Current y.
    ytmp0 : np.ndarray, shape (n,)
        ytmp = yinput - y at the current iterate.
    dxi : np.ndarray, shape (m,)
        Newton direction for ξ.
    Atdxi : np.ndarray, shape (n,)
        A^T @ dxi.
    tol : float
        Tolerance for sufficient decrease or curvature condition.
    options : dict
        Must contain 'stepop' (1 or 2).
    Returns
    -------
    par : dict
        Updated par dictionary (with updated 'rr').
    Ly : float
        New objective value at the updated iterate.
    xi : np.ndarray, shape (m,)
        Updated ξ = xi0 + α * dxi.
    Atxi : np.ndarray, shape (n,)
        A^T @ xi.
    y : np.ndarray, shape (n,)
        Updated y (projection onto L∞ ball).
    ytmp : np.ndarray, shape (n,)
        Updated ytmp = yinput - y.
    alp : float
        Step length α.
    iter : int
        Number of line-search iterations performed.
    """

    # Extract parameters
    stepop    = options.get('stepop', 1)
    sig       = par['sigma']

    # Precompute inner products and norms
    tmp1 = dxi.dot(b - xi0)                # dxi^T (b - xi0)
    tmp2 = np.linalg.norm(dxi)**2          # ||dxi||^2

    # Initial directional derivative at α = 0
    g0   = tmp1 + sig * Atdxi.dot(ytmp0)

    # If g0 <= 0, no ascent direction; return current iterates
    if g0 <= 0:
        alp   = 0.0
        iter_ = 0
        xi    = xi0.copy()
        Atxi  = Atxi0.copy()
        y     = y0.copy()
        ytmp  = ytmp0.copy()
        Ly    = Ly0
        return par, Ly, xi, Atxi, y, ytmp, alp, iter_

    # Initialize bounds for bi-section
    alp    = 1.0
    alp_lb = 0.0
    alp_ub = 1.0

    c1 = 1e-4
    c2 = 0.9

    Ly = None

    maxit = int(np.ceil(np.log(1.0 / (tol + np.finfo(float).eps)) / np.log(2.0)))

    for iter_ in range(1, maxit + 1):
        if iter_ == 1:
            alp = 1.0
            alp_lb = 0.0
            alp_ub = 1.0
        else:
            alp = 0.5 * (alp_lb + alp_ub)

        xi = xi0 + alp * dxi
        yinput = ytmp0 + y0 - alp * Atdxi
        y, rr = proj_inf(yinput, ld)   # project onto ∥y∥∞ ≤ ld
        par['rr'] = rr
        ytmp = yinput - y

        galp = tmp1 - alp * tmp2 + sig * Atdxi.dot(ytmp)

        # If sign(g0) = sign(galp) on first iteration, just accept α = 1
        if iter_ == 1:
            if g0 * galp > 0:
                Atxi = Atxi0 + alp * Atdxi
                Ly   = b.dot(xi) - 0.5 * np.linalg.norm(xi)**2 - 0.5 * sig * np.linalg.norm(ytmp)**2
                return par, Ly, xi, Atxi, y, ytmp, alp, iter_

            g_lb = g0
            g_ub = galp

        # Check curvature condition: |galp| ≤ c2 * |g0|
        if abs(galp) < c2 * abs(g0):
            Atxi_candidate = Atxi0 + alp * Atdxi
            Ly_candidate = b.dot(xi) - 0.5 * np.linalg.norm(xi)**2 - 0.5 * sig * np.linalg.norm(ytmp)**2
            if (Ly_candidate - Ly0 - c1 * alp * g0 > -1e-8 / max(1.0, abs(Ly0)) and
                ((stepop == 1) or (stepop == 2 and abs(galp) < tol))):
                Atxi = Atxi_candidate
                Ly   = Ly_candidate
                return par, Ly, xi, Atxi, y, ytmp, alp, iter_

        # Update bounds based on sign of galp
        if galp * g_ub < 0:
            alp_lb = alp
            g_lb   = galp
        elif galp * g_lb < 0:
            alp_ub = alp
            g_ub   = galp

    # If maximum iterations reached, accept last α
    Atxi = Atxi0 + alp * Atdxi
    if Ly is None:
        Ly = b.dot(xi) - 0.5 * np.linalg.norm(xi)**2 - 0.5 * sig * np.linalg.norm(ytmp)**2

    return par, Ly, xi, Atxi, y, ytmp, alp, iter_


# --------------------------------------------------------------------------
# Helper function: projection onto the ∞-norm ball of radius `ld`
# Returns y_proj = clamp(yinput, -ld, +ld), and rr = indicator array
# rr[i] = 1 if |yinput[i]| > ld (i.e., projection was active), else 0
# --------------------------------------------------------------------------
def proj_inf(yinput, ld):
    """
    Project yinput onto the ℓ∞-ball of radius ld: y = sign(yinput) * min(|yinput|, ld).
    Also return an indicator array rr where rr[i] = 1 if |yinput[i]| >= ld, else 0.
    """
    y = np.clip(yinput, -ld, ld)
    rr = (np.abs(yinput) >= ld).astype(int)
    return y, rr


# --------------------------------------------------------------------------
# Placeholder for the linear system solver used in SSNAL subproblem:
# In MATLAB, this is Classic_Lasso_linsys_solver(Ainput, rhs, par)
# In Python, you must implement or import an equivalent function that solves
#     H * dxi = rhs
# where H ≈ I + σ A A^T (or a preconditioned version) using PSQMR (or similar).
# It should return:
#   dxi    : ndarray, the computed Newton direction (shape (m,))
#   resnrm : list or array of residual norms (length = num_iters + 1)
#   solve_ok: integer flag indicating success/failure
# --------------------------------------------------------------------------
def classic_lasso_linsys_solver(Ainput, rhs, par):
    """
    Placeholder for PSQMR (or other Krylov solver) to solve:
        (I + σ A A^T) dxi = rhs
    or a preconditioned variant, as configured by `par`.

    Inputs
    ------
    Ainput : object with methods Amap, ATmap
    rhs    : np.ndarray, right-hand side vector (shape (m,))
    par    : dict, solver parameters, must contain:
                par['tol']   : tolerance for Krylov solver
                par['maxit'] : maximum # of Krylov iterations
                par['sigma'] : σ parameter
                par['precond']: 0 or 1 for using diagonal preconditioner

    Returns
    -------
    dxi     : np.ndarray, solution of shape (m,)
    resnrm  : list of float, residual norms at each iteration
    solve_ok: int, flag (0 if success, <0 if failure)
    """
    # === WARNING ===
    # This function must be implemented by the user (or imported from an existing module).
    # The implementation should solve (I + σ A A^T) dxi = rhs (m × m system),
    # possibly using PSQMR or MINRES with optional preconditioning.
    #
    # As a placeholder, we simply return:
    m = rhs.shape[0]
    dxi = np.zeros_like(rhs)
    resnrm = [0.0]  # pretend zero residual, one iteration
    solve_ok = 0    # indicate "success"
    return dxi, resnrm, solve_ok
