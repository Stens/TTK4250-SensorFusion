from typing import Tuple

import numpy as np
from numpy.core.fromnumeric import shape
from numpy.core.multiarray import where
import numpy.linalg as la
from scipy.stats.morestats import probplot


def discrete_bayes(
    # the prior: shape=(n,)
    pr: np.ndarray,
    # the conditional/likelihood: shape=(n, m)
    cond_pr: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray
]:  # the new marginal and conditional: shapes=((m,), (m, n))
    """Swap which discrete variable is the marginal and conditional."""

    joint =  cond_pr*(np.ones(cond_pr.shape[0])@pr.T) # TODO P(X,Y) = P(Y|X)P(X) 
    # joint = pr[:,None]*cond_pr

    marginal = np.sum(joint, axis=1)# TODO p(X) = sum of all x for  P(X,Y)

    # Take care of rare cases of degenerate zero marginal,
    conditional = np.divide(joint,marginal, where=True)

    # flip axes?? (n, m) -> (m, n)
    # conditional = conditional.T

    # optional DEBUG
    assert np.all(
        np.isfinite(conditional)
    ), f"NaN or inf in conditional in discrete bayes"
    assert np.all(
        np.less_equal(0, conditional)
    ), f"Negative values for conditional in discrete bayes"
    assert np.all(
        np.less_equal(conditional, 1)
    ), f"Value more than on in discrete bayes"

    assert np.all(np.isfinite(marginal)), f"NaN or inf in marginal in discrete bayes"

    return marginal, conditional
