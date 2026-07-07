import numpy as np
from fgrgeom import config as C
from fgrgeom import panel as P


class FactorAnalysisMissing:
    """Factor analysis x = W z + mu + e, e ~ N(0, diag(psi)), z ~ N(0, I_k),
    fit by EM with exact missing-data handling (observed dims only per sample,
    missing entries marginalised). NO imputation. Features standardised internally
    using observed-only moments."""

    def __init__(self, k, max_iter=300, tol=1e-4, seed=C.SEED):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def _standardize(self, X, M):
        mu = np.array([X[M[:, j], j].mean() if M[:, j].any() else 0.0
                       for j in range(X.shape[1])])
        sd = np.array([X[M[:, j], j].std() if M[:, j].sum() > 1 else 1.0
                       for j in range(X.shape[1])])
        sd[sd == 0] = 1.0
        self.center_, self.scale_ = mu, sd
        Xs = (X - mu) / sd
        Xs[~M] = 0.0
        return Xs

    def fit(self, X, M):
        rng = np.random.default_rng(self.seed)
        Xs = self._standardize(X, M)
        n, d = Xs.shape
        k = self.k
        W = rng.normal(scale=0.1, size=(d, k))
        psi = np.ones(d)
        mu = np.zeros(d)  # data already centered; mu absorbs residual offset
        ll_prev = -np.inf
        self.ll_trace_ = []

        for it in range(self.max_iter):
            Ez = np.zeros((n, k))
            Ezz = np.zeros((n, k, k))
            ll = 0.0
            for i in range(n):
                o = M[i]
                if not o.any():
                    Ezz[i] = np.eye(k)
                    continue
                Wo = W[o]
                psio = psi[o]
                xo = Xs[i, o] - mu[o]
                WtP = Wo.T / psio                      # (k, |o|)
                prec = np.eye(k) + WtP @ Wo            # (k,k)
                cov = np.linalg.inv(prec)
                ez = cov @ (WtP @ xo)
                Ez[i] = ez
                Ezz[i] = cov + np.outer(ez, ez)
                # observed-data log-likelihood: N(xo | 0, Wo Wo^T + diag(psio))
                Sig = Wo @ Wo.T + np.diag(psio)
                sign, logdet = np.linalg.slogdet(Sig)
                ll += -0.5 * (logdet + xo @ np.linalg.solve(Sig, xo)
                              + o.sum() * np.log(2 * np.pi))
            self.ll_trace_.append(ll)

            # M-step, per feature j over samples where j observed.
            for j in range(d):
                obs = M[:, j]
                if not obs.any():
                    continue
                xj = Xs[obs, j]
                Ez_j = Ez[obs]
                Ezz_j = Ezz[obs].sum(axis=0)
                rhs = (xj[:, None] * Ez_j).sum(axis=0)
                W[j] = np.linalg.solve(Ezz_j + 1e-6 * np.eye(k), rhs)
                mu[j] = (xj - Ez_j @ W[j]).mean()
                # psi_j = E[(x - mu - w^T z)^2] including posterior covariance of z
                covmean = (Ezz[obs] - np.einsum("ik,il->ikl",
                                                Ez[obs], Ez[obs])).mean(0)
                r = (xj - mu[j]) - Ez_j @ W[j]
                psi[j] = max((r ** 2).mean() + W[j] @ covmean @ W[j], 1e-4)

            if it > 0 and abs(ll - ll_prev) < self.tol * max(1.0, abs(ll_prev)):
                break
            ll_prev = ll

        self.W_, self.psi_, self.mu_ = W, psi, mu
        self.n_iter_ = it + 1
        self.Ez_, self.Ezz_ = Ez, Ezz
        return self

    def transform(self, X, M):
        Xs = (X - self.center_) / self.scale_
        Xs[~M] = 0.0
        n, k = X.shape[0], self.k
        Ez = np.zeros((n, k))
        Cov = np.zeros((n, k, k))
        for i in range(n):
            o = M[i]
            if not o.any():
                Cov[i] = np.eye(k)
                continue
            Wo = self.W_[o]
            WtP = Wo.T / self.psi_[o]
            cov = np.linalg.inv(np.eye(k) + WtP @ Wo)
            Cov[i] = cov
            Ez[i] = cov @ (WtP @ (Xs[i, o] - self.mu_[o]))
        return Ez, Cov


def fit_latent(panel, k=6, include=("biom", "doppler", "maternal"), **kw):
    """Fit FactorAnalysisMissing on the flattened per-fetus panel.
    Returns dict: W (d,k), Z (n,k) posterior means, Zcov (n,k,k), psi (d,),
    colnames, model, and the input mask."""
    X, M, names = P.flatten(panel, include=include)
    fa = FactorAnalysisMissing(k=k, **kw).fit(X, M)
    Z, Zcov = fa.transform(X, M)
    return {"W": fa.W_, "Z": Z, "Zcov": Zcov, "psi": fa.psi_,
            "colnames": names, "model": fa, "mask": M,
            "ll": fa.ll_trace_[-1] if fa.ll_trace_ else None}
