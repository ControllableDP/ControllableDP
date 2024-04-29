
import numpy as np
import math

from scipy import special

# from utils.rdp_convert_dp import compute_eps
def compute_eps(orders, rdp, delta):
  """Compute epsilon given a list of RDP values and target delta.
  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    delta: The target delta.
  Returns:
    Pair of (eps, optimal_order).
  Raises:
    ValueError: If input is malformed.
  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if delta <= 0:
    raise ValueError("Privacy failure probability bound delta must be >0.")
  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")

  eps_vec = []
  for (a, r) in zip(orders_vec, rdp_vec):
    if a < 1:
      raise ValueError("Renyi divergence order must be >=1.")
    if r < 0:
      raise ValueError("Renyi divergence must be >=0.")

    if delta**2 + math.expm1(-r) >= 0:
      eps = 0
    elif a > 1.01:
      eps = ( r - (np.log(delta) + np.log(a)) / (a - 1) + np.log((a - 1) / a))
    else:
      eps = np.inf
    eps_vec.append(eps)


  idx_opt = np.argmin(eps_vec)
  return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]


def compute_eps2(orders, rdp, delta):
  """Compute epsilon given a list of RDP values and target delta.
  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    delta: The target delta.
  Returns:
    Pair of (eps, optimal_order).
  Raises:
    ValueError: If input is malformed.
  """
  r"""Computes epsilon given a list of Renyi Differential Privacy (RDP) values at
  multiple RDP orders and target ``delta``.

  Args:
      orders: An array (or a scalar) of orders (alphas).
      rdp: A list (or a scalar) of RDP guarantees.
      delta: The target delta.

  Returns:
      Pair of epsilon and optimal order alpha.

  Raises:
      ValueError
          If the lengths of ``orders`` and ``rdp`` are not equal.
  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if len(orders_vec) != len(rdp_vec):
    raise ValueError(
      f"Input lists must have the same length.\n"
      f"\torders_vec = {orders_vec}\n"
      f"\trdp_vec = {rdp_vec}\n"
    )

  eps = rdp_vec - math.log(delta) / (orders_vec - 1)

  if np.isnan(eps).all():
    return np.inf, np.nan

  idx_opt = np.nanargmin(eps)
  return eps[idx_opt], orders_vec[idx_opt]

def compute_rdp(q, noise_multiplier, n_steps, orders):
  """Computes RDP of the Sampled Gaussian Mechanism.
  Args:
    q: The sampling rate.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise    STD标准差，敏感度应该包含在这里面了
      to the l2-sensitivity of the function to which it is added.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.
  Returns:
    The RDPs at all orders. Can be `np.inf`.
  """
  if np.isscalar(orders):
    rdp = _compute_rdp(q, noise_multiplier, orders)
  else:
    rdp = np.array(
        [ _compute_rdp(q, noise_multiplier, order) for order in orders])
  return (rdp * n_steps).reshape(-1)

def _compute_log_a_for_int_alpha(q, sigma, alpha):
    assert isinstance(alpha, int)
    rdp = -np.inf

    for i in range(alpha + 1):
        log_b = (
                math.log(special.binom(alpha, i))
                + i * math.log(q)
                + (alpha - i) * math.log(1 - q)
                + (i * i - i) / (2 * (sigma ** 2))
        )


        a, b = min(rdp, log_b), max(rdp, log_b)
        if a == -np.inf:  # adding 0
            rdp = b
        else:
            rdp = math.log(math.exp(
                a - b) + 1) + b

    rdp = float(rdp) / (alpha - 1)
    return rdp


def _log_add(logx: float, logy: float) -> float:
    r"""Adds two numbers in the log space.

    Args:
        logx: First term in log space.
        logy: Second term in log space.

    Returns:
        Sum of numbers in log space.
    """
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:
        return b
    return math.log1p(math.exp(a - b)) + b


def _log_sub(logx: float, logy: float) -> float:
    r"""Subtracts two numbers in the log space.

    Args:
        logx: First term in log space. Expected to be greater than the second term.
        logy: First term in log space. Expected to be less than the first term.

    Returns:
        Difference of numbers in log space.

    Raises:
        ValueError
            If the result is negative.
    """
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        return math.log(math.expm1(logx - logy)) + logy
    except OverflowError:
        return logx

def _log_erfc(x: float) -> float:
    r"""Computes :math:`log(erfc(x))` with high accuracy for large ``x``.

    Helper function used in computation of :math:`log(A_\alpha)`
    for a fractional alpha.

    Args:
        x: The input to the function

    Returns:
        :math:`log(erfc(x))`
    """
    return math.log(2) + special.log_ndtr(-x * 2 ** 0.5)

def _compute_log_a_for_frac_alpha(q: float, sigma: float, alpha: float) -> float:
    r"""Computes :math:`log(A_\alpha)` for fractional ``alpha``.

    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.

        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf for details.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        :math:`log(A_\alpha)` as defined in Section 3.3 of
        https://arxiv.org/pdf/1908.10530.pdf.
    """
    # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
    # initialized to 0 in the log space:
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0

    z0 = sigma ** 2 * math.log(1 / q - 1) + 0.5

    while True:  # do ... until loop
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * (sigma ** 2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma ** 2)) + log_e1

        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return _log_add(log_a0, log_a1) / (alpha - 1)

def _compute_rdp(q, sigma, alpha):
    """Compute RDP of the Sampled Gaussian mechanism at order alpha.
    Args:
      q: The sampling rate.
      sigma: The std of the additive Gaussian noise.
      alpha: The order at which RDP is computed.
    Returns:
      RDP at alpha, can be np.inf.

      q==1时的公式可参考：[renyi differential privacy,2017,Proposition 7]
      0<q<1时，有以下两个公式：
      可以参考[Renyi Differential Privacy of the Sampled Gaussian Mechanism ,2019,3.3]，这篇文章中包括alpha为浮点数的计算
      公式2更为简洁的表达在[User-Level Privacy-Preserving Federated Learning: Analysis and Performance Optimization,2021,3.2和3.3]
    """
    if q == 0:
        return 0

    # no privacy
    if sigma == 0:
        return np.inf

    if q == 1.:
        return alpha  / (2 * sigma ** 2)

    if np.isinf(alpha):
        return np.inf

    if float(alpha).is_integer():
        return _compute_log_a_for_int_alpha(q, sigma, int(alpha))
    else:
        return _compute_log_a_for_frac_alpha(q, sigma, alpha)

def compute_rdp_randomized_response(p,steps,orders):

    if np.isscalar(orders):
        rdp = _compute_rdp_randomized_response(p, orders)
    else:
        rdp = np.array([_compute_rdp_randomized_response(p, order) for order in orders])

    return rdp * steps

def _compute_rdp_randomized_response(p,alpha):
    from decimal import Decimal
    a=Decimal(p**alpha)
    b=Decimal((1-p)**(1-alpha))
    c=Decimal(a*b)
    item1=float((p**alpha)*((1-p)**(1-alpha)))
    item2=float(((1-p)**alpha)*(p**(1-alpha)))
    rdp=float(math.log(item1+item2))/ (alpha-1)
    return rdp


# if __name__ == '__main__':
#     ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(2, 64)) + [128]

#     q=3000/60000
#     rdp = compute_rdp_randomized_response(0.99,1, ORDERS)
#     dp,order=compute_eps(ORDERS,rdp,1e-5)
#     print("rdp:",rdp*q)
#     print("dp:",dp*q)
#     print("order:",order)

#     ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(2, 64)) + [128]

#     rdp = compute_rdp(256/60000, 1.23, 1, ORDERS)
#     dp,order=compute_eps(ORDERS,rdp,1e-5)
#     print("rdp:",rdp)
#     print("dp:",dp)
#     print("order:",order)

