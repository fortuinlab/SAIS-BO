
import torch
from torch import Tensor
import pygmo as pg
import numpy as np


from botorch.models.model import Model

from botorch.acquisition import AnalyticAcquisitionFunction, LogExpectedImprovement, UpperConfidenceBound

from botorch.optim import optimize_acqf

from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths


dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PosteriorSample(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
    ) -> None:
        super(AnalyticAcquisitionFunction, self).__init__(model)

        self.path = draw_matheron_paths(self.model, torch.Size([1]))

    def forward(self, X: Tensor):
        
        # X: N, ..., d

        y = self.path(X)

        return y.flatten() 

class PenalizedUpperConfidenceBound(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: Tensor,
        bounds: Tensor,
        busy: Tensor | None = None,
        y_max = None,
        local = False
    ) -> None:
        super(AnalyticAcquisitionFunction, self).__init__(model)

        self.register_buffer("beta", torch.as_tensor(beta, dtype=dtype, device = device))
        self.register_buffer("bounds", torch.as_tensor(bounds, dtype=dtype, device = device))
        if y_max is not None:
            self.register_buffer("y_max", torch.as_tensor(y_max, dtype=dtype, device = device))
        
        if busy is not None:

            self.register_buffer("busy", torch.as_tensor(busy, dtype=dtype, device = device))
            grad_norm = AnalyticPostMeanGradientNorm(self.model)

            d = self.bounds.shape[-1]

            if local:
                ls = self.model.covar_module.lengthscale 

                bounds_l = torch.clamp_min(self.busy-ls, self.bounds[0])
                bounds_u = torch.clamp_max(self.busy+ls, self.bounds[1])
                bounds_ = torch.stack((bounds_l, bounds_u), dim = 1)
    
                L = []
                norm_maxer = []
                for i in range(len(self.busy)):
                    
                    maxer, l = optimize_acqf(   acq_function=grad_norm,
                                                bounds=bounds_[i],
                                                q=1,
                                                num_restarts=10,
                                                raw_samples=d*1000,
                                                options={"batch_limit": 50, "maxiter": 200},
                                            )

                    L.append(l)
                    norm_maxer.append(maxer)

                L = torch.tensor(L, dtype=dtype, device = device).reshape(1,len(self.busy))
                norm_maxer = torch.cat(norm_maxer).reshape(-1, d) 
            else:

                norm_maxer, L = optimize_acqf(  acq_function=grad_norm,
                                                bounds=self.bounds,
                                                q=1,
                                                num_restarts=10,
                                                raw_samples=d*1000,
                                                options={"batch_limit": 50, "maxiter": 200},
                                            )
                
                L = L.to(dtype=dtype, device = device).reshape(1,1)

            self.register_buffer("L", torch.as_tensor(L, dtype=dtype, device=device))
            self.register_buffer("norm_maxer", torch.as_tensor(norm_maxer, dtype=dtype))
            
        else:
            self.busy = busy

        
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X (tensor): Nxq=1xd

        Returns:
            acqf value (tensor): N
        """

        p = -5
        
        posterior = self.model.posterior(X)
        mean = posterior.mean
        std = posterior.variance.sqrt()
        ucb =  (mean + self.beta.sqrt() * std).flatten()

        if self.busy is None:
            return ucb 

        post_b = self.model.posterior(self.busy)
        mean_b = post_b.mean
   
        eps = 1e-8
        std_b = post_b.variance.sqrt()

        if len(X.shape) == 3: X = X.squeeze(1)
        norm = torch.cdist(X, self.busy) 

        s = ((torch.abs(mean_b - self.y_max) + 1 * std_b)).reshape(1,-1) / (self.L) 

        weights = norm / s 

        diff_weights = (weights**p + 1)**(1/p)

        pen = torch.sum(torch.log(diff_weights), dim=1) 

        pen_ucb = torch.exp(torch.log(ucb.clamp_min(eps)) + pen)

        return pen_ucb 
    
class AnalyticPostMeanGradientNorm(AnalyticAcquisitionFunction):
    def __init__(self, 
                 model: Model,
                 ) -> None:
        super().__init__(model)

        self.k = model.covar_module
        self.Theta_inv = torch.atleast_2d(torch.diag(1/self.k.lengthscale.flatten()**2))
        self.train_X = model.input_transform.untransform(model.train_inputs[0])
        self.train_Y = model.outcome_transform.untransform(model.train_targets)[0].reshape(-1,1)

        K_X_X = self.k(self.train_X).evaluate()
        sig_squ = model.likelihood.noise

        K_noise = K_X_X + (sig_squ + 1e-8) * torch.eye(K_X_X.size(0), dtype=dtype, device=device)

        L = torch.linalg.cholesky(K_noise + 1e-8 * torch.eye(K_X_X.size(0), dtype=dtype, device=device))
        K_noise_inv = torch.cholesky_inverse(L)

        self.K_noise_inv_Y = torch.matmul(K_noise_inv, self.train_Y)
        
    def forward(self, X: Tensor) -> Tensor:
        
        # X: N,d
        if len(X.shape) == 3: X = X.squeeze(1)

        K_st_X = self.k(X, self.train_X).evaluate().unsqueeze(-1)
        D = (self.train_X.unsqueeze(0)-X.unsqueeze(1))
        grad_K_st_X = K_st_X * D @ self.Theta_inv

        dmu_dx = torch.linalg.matmul(grad_K_st_X.transpose(1,2),
                                     self.K_noise_inv_Y).squeeze(-1)

        grad_norm = torch.linalg.vector_norm(dmu_dx, dim=-1)
        return grad_norm.clamp_min(1e-8) # N

# The following function is taken directly from the authors of AEGiS
# https://github.com/georgedeath/aegis/blob/main/aegis/batch/nsga2_pygo.py

def NSGA2_pygmo(model, fevals, lb, ub, cf=None):
    """Finds the estimated Pareto front of a gpytorch model using NSGA2 [1]_.

    Parameters
    ----------
    model: gpytorch.models.ExactGP
        gpytorch regression model on which to find the Pareto front
        of its mean prediction and standard deviation.
    fevals : int
        Maximum number of times to evaluate a location using the model.
    lb : (D, ) torch.tensor
        Lower bound box constraint on D
    ub : (D, ) torch.tensor
        Upper bound box constraint on D
    cf : callable, optional
        Constraint function that returns True if it is called with a
        valid decision vector, else False.

    Returns
    -------
    X_front : (F, D) numpy.ndarray
        The F D-dimensional locations on the estimated Pareto front.
    musigma_front : (F, 2) numpy.ndarray
        The corresponding mean response and standard deviation of the locations
        on the front such that a point X_front[i, :] has a mean prediction
        musigma_front[i, 0] and standard deviation musigma_front[i, 1].

    Notes
    -----
    NSGA2 [1]_ discards locations on the pareto front if the size of the front
    is greater than that of the population size. We counteract this by storing
    every location and its corresponding mean and standard deviation and
    calculate the Pareto front from this - thereby making the most of every
    GP model evaluation.

    References
    ----------
    .. [1] Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T. Meyarivan.
       A fast and elitist multiobjective genetic algorithm: NSGA-II.
       IEEE Transactions on Evolutionary Computation 6, 2 (2001), 182–197.
    """
    # internal class for the pygmo optimiser
    class GPYTORCH_WRAPPER(object):
        def __init__(self, model, lb, ub, cf, evals):
            # model = gpytorch model
            # lb = torch.tensor of lower bounds on X
            # ub = torch.tensor of upper bounds on X
            # cf = callable constraint function
            # evals = total evaluations to be carried out
            self.model = model
            self.lb = lb.numpy()
            self.ub = ub.numpy()
            self.nd = lb.numel()
            self.got_cf = cf is not None
            self.cf = cf
            self.i = 0  # evaluation pointer
            self.dtype = model.train_targets.dtype

        def get_bounds(self):
            return (self.lb, self.ub)

        def get_nobj(self):
            return 2

        def fitness(self, X):
            X = np.atleast_2d(X)
            X = torch.as_tensor(X, dtype=self.dtype)

            f = model_fitness(
                X,
                self.model,
                self.cf,
                self.got_cf,
                self.i,
                self.i + X.shape[0],
            )

            self.i += X.shape[0]
            return f.ravel()

        def has_batch_fitness(self):
            return True

        def batch_fitness(self, X):
            X = X.reshape(-1, self.nd)
            return self.fitness(X)

    # fitness function for the optimiser
    def model_fitness(X, model, cf, got_cf, start_slice, end_slice):
        n = X.shape[0]

        f = np.zeros((n, 2))
        valid_mask = np.ones(n, dtype="bool")

        # if we select a location that violates the constraint,
        # ensure it cannot dominate anything by having its fitness values
        # maximally bad (i.e. set to infinity)
        if got_cf:
            for i in range(n):
                if not cf(X[i]):
                    f[i] = [np.inf, np.inf]
                    valid_mask[i] = False

        if np.any(valid_mask):
            output = model(X[valid_mask])
            output = model.likelihood(
                output,
                noise=torch.full_like(
                    output.mean, model.likelihood.noise.mean()
                ),
            )

            # note the negative stdev here as NSGA2 is minimising
            # so we want to minimise the negative stdev
            f[valid_mask, 0] = output.mean.numpy()
            f[valid_mask, 1] = -np.sqrt(output.variance.numpy())

        # store every location ever evaluated
        model_fitness.X[start_slice:end_slice, :] = X
        model_fitness.Y[start_slice:end_slice, :] = f

        return f

    # get the problem dimensionality
    D = lb.numel()

    # NSGA-II settings
    POPSIZE = D * 100
    # -1 here because the pop is evaluated first before iterating N_GENS times
    N_GENS = int(np.ceil(fevals / POPSIZE)) - 1
    TOTAL_EVALUATIONS = POPSIZE * (N_GENS + 1)

    _nsga2 = pg.nsga2(
        gen=1,  # number of generations to evaluate per evolve() call
        cr=0.8,  # cross-over probability.
        eta_c=20.0,  # distribution index (cr)
        m=1 / D,  # mutation rate
        eta_m=20.0,  # distribution index (m)
    )

    # batch fitness evaluator -- this is the strange way we
    # tell pygmo that we have a batch_fitness method
    bfe = pg.bfe()

    # tell nsgaII about it
    _nsga2.set_bfe(bfe)
    nsga2 = pg.algorithm(_nsga2)

    # preallocate the storage of every location and fitness to be evaluated
    model_fitness.X = np.zeros((TOTAL_EVALUATIONS, D))
    model_fitness.Y = np.zeros((TOTAL_EVALUATIONS, 2))

    # problem instance
    gpytorch_problem = GPYTORCH_WRAPPER(model, lb, ub, cf, TOTAL_EVALUATIONS)
    problem = pg.problem(gpytorch_problem)

    # skip all gradient calculations as we don't need them
    with torch.no_grad():
        # initialise the population -- in batch (using bfe)
        population = pg.population(problem, size=POPSIZE, b=bfe)

        # evolve the population
        for i in range(N_GENS):
            population = nsga2.evolve(population)

    # indices non-dominated points across the entire NSGA-II run
    front_inds = pg.non_dominated_front_2d(model_fitness.Y)

    X_front = model_fitness.X[front_inds, :]
    musigma_front = model_fitness.Y[front_inds, :]

    # convert the standard deviations back to positive values; nsga2 minimises
    # the negative standard deviation (i.e. maximises the standard deviation)
    musigma_front[:, 1] *= -1

    # convert it to torch
    X_front = torch.as_tensor(X_front, dtype=model.train_targets.dtype)
    musigma_front = torch.as_tensor(
        musigma_front, dtype=model.train_targets.dtype
    )

    return X_front, musigma_front

class AEGIS(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        bounds: Tensor,
    ) -> None:
        super(AnalyticAcquisitionFunction, self).__init__(model)

        d = torch.tensor(len(model.covar_module.lengthscale[0]))

        eps = torch.min(1.0 / torch.sqrt(d), torch.tensor(0.5))

        r = torch.rand(1)
        if r < 1 - (eps + eps):
            # exploit
            self.mode = "exploit"

        elif r < 1 - eps:
            # Thompson
            self.mode = "Thompson"
            self.path = draw_matheron_paths(self.model, torch.Size([1]))
            
        else:
            # approx Pareto selection
            self.mode = "Pareto"

            self.pareto_front,  _ = NSGA2_pygmo(
                model=model, fevals=1, lb=bounds[0], ub=bounds[1], cf=None
            )

        self.model = model

    def forward(self, X):

        if self.mode == "exploit":

            post = self.model.posterior(X)

            y = post.mean.squeeze(-1)

            return y.flatten()
        
        elif self.mode == "Thompson":

            y = self.path(X)

            return y.flatten()

        
class UCB_KB(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        B: Tensor | None,
    ) -> None:
        super(AnalyticAcquisitionFunction, self).__init__(model)

        if B is not None:
            with torch.no_grad():
                self.B = self.model.input_transform(B).detach() 
                self.post = model.posterior(B) 
                self.mu_B = self.post.mean

            self.gp = self.model.condition_on_observations(self.B, self.mu_B)

        else:
            self.B = B
        
        self.beta = 2

    def forward(self, x: Tensor) -> Tensor:
        
        """
        Input:
            x: (m,d)
        """

        if self.B is None:

            post = self.model.posterior(x)
            mu = post.mean
            std = post.variance.sqrt()

            ucb = mu + self.beta**0.5 * std

            return ucb

        if x.ndimension() == 2:
            x = x.unsqueeze(0)


        ucb_kb = UpperConfidenceBound(self.gp, self.beta)

        return ucb_kb(x)


class LogEI_KB(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        B: Tensor | None,
    ) -> None:
        super(AnalyticAcquisitionFunction, self).__init__(model)

        Y = model.train_targets.unsqueeze(-1)
        mean = model.outcome_transform.means
        std = model.outcome_transform.stdvs
        self.Y = Y * std + mean 
        
        if B is not None:
            with torch.no_grad():
                self.B = self.model.input_transform(B).detach() 
                self.post = model.posterior(B) 
                self.mu_B = self.post.mean 

            self.gp = self.model.condition_on_observations(self.B, self.mu_B)
            self.y_best = torch.cat((self.Y, self.mu_B), dim=0).max()
        else:
            self.B = B
            self.y_best = self.Y.max()
        
        self.beta = 2

    def forward(self, x: Tensor) -> Tensor:
        
        """
        Input:
            x: (m,d)
        """

        if self.B is None:

            
            logei = LogExpectedImprovement(self.model, self.y_best)

            return logei(x)

        if x.ndimension() == 2:
            x = x.unsqueeze(0)


        logei_kb = LogExpectedImprovement(self.gp, self.y_best)

        return logei_kb(x)









