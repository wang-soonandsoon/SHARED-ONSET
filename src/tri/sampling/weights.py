"""Path-importance identity, independent of any resampling implementation."""
import math

from tri.errors import InvalidSpecification


def log_path_increment(*, log_rho_reference: float, log_rho_proposal: float,
                       log_q_batch: float, log_z_next: float,
                       log_z_clamped: float) -> float:
    """log G = log(rho_ref/rho_prop) + log q_A + log Z_new - log Z_old,clamped.

    All arguments except Z_next must be finite for a sampled supported edge.
    Z_next=-inf gives zero weight; it never causes an automatic restart.
    """
    terms = (log_rho_reference, log_rho_proposal, log_q_batch, log_z_clamped)
    if any(not math.isfinite(v) for v in terms) or math.isnan(log_z_next) or log_z_next == math.inf:
        raise InvalidSpecification("invalid log values on a sampled path transition")
    return log_rho_reference - log_rho_proposal + log_q_batch + log_z_next - log_z_clamped
