import warnings
from copy import copy

import numpy as np
from poap.strategy import EvalRecord
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.optimization_problems import OptimizationProblem
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant, SurrogateUnitBox

from hyperopt import hp, tpe
from hyperopt.base import JOB_STATE_DONE, JOB_STATE_NEW, STATUS_OK, Domain, Trials
from scipy.interpolate import interp1d

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.space import JointSpace
from bayesmark.np_util import random as np_random
from bayesmark.np_util import random_seed


class PySOTOptimizer(AbstractOptimizer):
    primary_import = "pysot"

    def __init__(self, api_config):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)

        self.space_x = JointSpace(api_config)
        self.bounds = self.space_x.get_bounds()
        self.create_opt_prob()  # Sets up the optimization problem (needs self.bounds)
        self.max_evals = np.iinfo(np.int32).max  # NOTE: Largest possible int
        self.batch_size = None
        self.history = []
        self.proposals = []

    def create_opt_prob(self):
        """Create an optimization problem object."""
        opt = OptimizationProblem()
        opt.lb = self.bounds[:, 0]  # In warped space
        opt.ub = self.bounds[:, 1]  # In warped space
        opt.dim = len(self.bounds)
        opt.cont_var = np.arange(len(self.bounds))
        opt.int_var = []
        assert len(opt.cont_var) + len(opt.int_var) == opt.dim
        opt.objfun = None
        self.opt = opt

    def start(self, max_evals):
        """Starts a new pySOT run."""
        self.history = []
        self.proposals = []

        # Symmetric Latin hypercube design
        des_pts = max([self.batch_size, 2 * (self.opt.dim + 1)])
        slhd = SymmetricLatinHypercube(dim=self.opt.dim, num_pts=des_pts)

        # Warped RBF interpolant
        rbf = RBFInterpolant(dim=self.opt.dim, kernel=CubicKernel(), tail=LinearTail(self.opt.dim), eta=1e-4)
        rbf = SurrogateUnitBox(rbf, lb=self.opt.lb, ub=self.opt.ub)

        # Optimization strategy
        self.strategy = SRBFStrategy(
            max_evals=self.max_evals,
            opt_prob=self.opt,
            exp_design=slhd,
            surrogate=rbf,
            asynchronous=True,
            batch_size=1,
            use_restarts=True,
        )

    def suggest(self, n_suggestions=1):
        """Get a suggestion from the optimizer.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """

        if self.batch_size is None:  # First call to suggest
            self.batch_size = n_suggestions
            self.start(self.max_evals)

        # Set the tolerances pretending like we are running batch
        d, p = float(self.opt.dim), float(n_suggestions)
        self.strategy.failtol = p * int(max(np.ceil(d / p), np.ceil(4 / p)))

        # Now we can make suggestions
        x_w = []
        self.proposals = []
        for _ in range(n_suggestions):
            proposal = self.strategy.propose_action()
            record = EvalRecord(proposal.args, status="pending")
            proposal.record = record
            proposal.accept()  # This triggers all the callbacks

            # It is possible that pySOT proposes a previously evaluated point
            # when all variables are integers, so we just abort in this case
            # since we have likely converged anyway. See PySOT issue #30.
            x = list(proposal.record.params)  # From tuple to list
            x_unwarped, = self.space_x.unwarp(x)
            if x_unwarped in self.history:
                warnings.warn("pySOT proposed the same point twice")
                self.start(self.max_evals)
                return self.suggest(n_suggestions=n_suggestions)

            # NOTE: Append unwarped to avoid rounding issues
            self.history.append(copy(x_unwarped))
            self.proposals.append(proposal)
            x_w.append(copy(x_unwarped))

        return x_w

    def _observe(self, x, y):
        # Find the matching proposal and execute its callbacks
        idx = [x == xx for xx in self.history]
        i = np.argwhere(idx)[0].item()  # Pick the first index if there are ties
        proposal = self.proposals[i]
        proposal.record.complete(y)
        self.proposals.pop(i)
        self.history.pop(i)

    def observe(self, X, y):
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        assert len(X) == len(y)

        for x_, y_ in zip(X, y):
            # Just ignore, any inf observations we got, unclear if right thing
            if np.isfinite(y_):
                self._observe(x_, y_)

# Sklearn prefers str to unicode:
DTYPE_MAP = {"real": float, "int": int, "bool": bool, "cat": str, "ordinal": str}


def dummy_f(x):
    assert False, "This is a placeholder, it should never be called."


def only(x):
    y, = x
    return y


class HyperoptOptimizer(AbstractOptimizer):
    primary_import = "hyperopt"

    def __init__(self, api_config, random=np_random):
        """Build wrapper class to use hyperopt optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)
        self.random = random

        space, self.round_to_values = HyperoptOptimizer.get_hyperopt_dimensions(api_config)
        self.domain = Domain(dummy_f, space, pass_expr_memo_ctrl=None)
        self.trials = Trials()

        # Some book keeping like opentuner wrapper
        self.trial_id_lookup = {}

        # Store just for data validation
        self.param_set_chk = frozenset(api_config.keys())

    @staticmethod
    def hashable_dict(d):
        """A custom function for hashing dictionaries.

        Parameters
        ----------
        d : dict or dict-like
            The dictionary to be converted to immutable/hashable type.

        Returns
        -------
        hashable_object : frozenset of tuple pairs
            Bijective equivalent to dict that can be hashed.
        """
        hashable_object = frozenset(d.items())
        return hashable_object

    @staticmethod
    def get_hyperopt_dimensions(api_config):
        """Help routine to setup hyperopt search space in constructor.

        Take api_config as argument so this can be static.
        """
        # The ordering of iteration prob makes no difference, but just to be
        # safe and consistnent with space.py, I will make sorted.
        param_list = sorted(api_config.keys())

        space = {}
        round_to_values = {}
        for param_name in param_list:
            param_config = api_config[param_name]

            param_type = param_config["type"]

            param_space = param_config.get("space", None)
            param_range = param_config.get("range", None)
            param_values = param_config.get("values", None)

            # Some setup for case that whitelist of values is provided:
            values_only_type = param_type in ("cat", "ordinal")
            if (param_values is not None) and (not values_only_type):
                assert param_range is None
                param_values = np.unique(param_values)
                param_range = (param_values[0], param_values[-1])
                round_to_values[param_name] = interp1d(
                    param_values, param_values, kind="nearest", fill_value="extrapolate"
                )

            if param_type == "int":
                low, high = param_range
                if param_space in ("log", "logit"):
                    space[param_name] = hp.qloguniform(param_name, np.log(low), np.log(high), 1)
                else:
                    space[param_name] = hp.quniform(param_name, low, high, 1)
            elif param_type == "bool":
                assert param_range is None
                assert param_values is None
                space[param_name] = hp.choice(param_name, (False, True))
            elif param_type in ("cat", "ordinal"):
                assert param_range is None
                space[param_name] = hp.choice(param_name, param_values)
            elif param_type == "real":
                low, high = param_range
                if param_space in ("log", "logit"):
                    space[param_name] = hp.loguniform(param_name, np.log(low), np.log(high))
                else:
                    space[param_name] = hp.uniform(param_name, low, high)
            else:
                assert False, "type %s not handled in API" % param_type

        return space, round_to_values

    def get_trial(self, trial_id):
        for trial in self.trials._dynamic_trials:
            if trial["tid"] == trial_id:
                assert isinstance(trial, dict)
                # Make sure right kind of dict
                assert "state" in trial and "result" in trial
                assert trial["state"] == JOB_STATE_NEW
                return trial
        assert False, "No matching trial ID"

    def cleanup_guess(self, x_guess):
        assert isinstance(x_guess, dict)
        # Also, check the keys are only the vars we are searching over:
        assert frozenset(x_guess.keys()) == self.param_set_chk

        # Do the rounding
        # Make a copy to be safe, and also unpack singletons
        # We may also need to consider clip_chk at some point like opentuner
        x_guess = {k: only(x_guess[k]) for k in x_guess}
        for param_name, round_f in self.round_to_values.items():
            x_guess[param_name] = round_f(x_guess[param_name])
        # Also ensure this is correct dtype so sklearn is happy
        x_guess = {k: DTYPE_MAP[self.api_config[k]["type"]](x_guess[k]) for k in x_guess}
        return x_guess

    def _suggest(self):
        """Helper function to `suggest` that does the work of calling
        `hyperopt` via its dumb API.
        """
        new_ids = self.trials.new_trial_ids(1)
        assert len(new_ids) == 1
        self.trials.refresh()

        seed = random_seed(self.random)
        new_trials = tpe.suggest(new_ids, self.domain, self.trials, seed)
        assert len(new_trials) == 1

        self.trials.insert_trial_docs(new_trials)
        self.trials.refresh()

        new_trial, = new_trials  # extract singleton
        return new_trial

    def suggest(self, n_suggestions=1):
        """Make `n_suggestions` suggestions for what to evaluate next.

        This requires the user observe all previous suggestions before calling
        again.

        Parameters
        ----------
        n_suggestions : int
            The number of suggestions to return.

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        assert n_suggestions >= 1, "invalid value for n_suggestions"

        # Get the new trials, it seems hyperopt either uses random search or
        # guesses one at a time anyway, so we might as welll call serially.
        new_trials = [self._suggest() for _ in range(n_suggestions)]

        X = []
        for trial in new_trials:
            x_guess = self.cleanup_guess(trial["misc"]["vals"])
            X.append(x_guess)

            # Build lookup to get original trial object
            x_guess_ = HyperoptOptimizer.hashable_dict(x_guess)
            assert x_guess_ not in self.trial_id_lookup, "the suggestions should not already be in the trial dict"
            self.trial_id_lookup[x_guess_] = trial["tid"]

        assert len(X) == n_suggestions
        return X

    def observe(self, X, y):
        """Feed the observations back to hyperopt.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated.
        """
        assert len(X) == len(y)

        for x_guess, y_ in zip(X, y):
            x_guess_ = HyperoptOptimizer.hashable_dict(x_guess)
            assert x_guess_ in self.trial_id_lookup, "Appears to be guess that did not originate from suggest"

            assert x_guess_ in self.trial_id_lookup, "trial object not available in trial dict"
            trial_id = self.trial_id_lookup.pop(x_guess_)
            trial = self.get_trial(trial_id)
            assert self.cleanup_guess(trial["misc"]["vals"]) == x_guess, "trial ID not consistent with x values stored"

            # Cast to float to ensure native type
            result = {"loss": float(y_), "status": STATUS_OK}
            trial["state"] = JOB_STATE_DONE
            trial["result"] = result
        # hyperopt.fmin.FMinIter.serial_evaluate only does one refresh at end
        # of loop of a bunch of evals, so we will do the same thing here.
        self.trials.refresh()
