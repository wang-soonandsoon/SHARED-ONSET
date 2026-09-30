"""Small, exact arithmetic oracle for TRI v7.1, not a music model.

No network or third-party dependencies. Enumerates a three-bit masked process
and checks local proposal probabilities against a fixed reference path target.
The q table is newly constructed for this implementation guide; this is NOT a
reproduction of the unspecified numerical table in v7.1 Appendix B.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
from typing import Iterable

State = tuple[int | None, ...]
N = 3
T = 3
INITIAL: State = (None,) * N


def multiply(values: Iterable[F]) -> F:
    out = F(1)
    for value in values:
        out *= value
    return out


def missing(x: State) -> tuple[int, ...]:
    return tuple(i for i, value in enumerate(x) if value is None)


def subsets(items: tuple[int, ...]):
    for size in range(len(items) + 1):
        yield from combinations(items, size)


def pin(x: State, A: tuple[int, ...], a: tuple[int, ...]) -> State:
    if len(A) != len(a) or len(set(A)) != len(A):
        raise ValueError('Invalid batch assignment')
    y = list(x)
    for i, value in zip(A, a):
        if i not in missing(x) or value not in (0, 1):
            raise ValueError('Only missing binary positions can be revealed')
        y[i] = value
    return tuple(y)


def q1(t: int, x: State, i: int) -> F:
    """Positive rational probabilities that change with state AND time."""
    context = sum((j + 1) * (v + 1) for j, v in enumerate(x) if v is not None)
    return F(1 + ((2 * t + 3 * i + context) % 7), 8)


def q_complete(t: int, x: State, y: tuple[int, ...]) -> F:
    if any(v is not None and y[i] != v for i, v in enumerate(x)):
        return F(0)
    return multiply(q1(t, x, i) if y[i] else 1 - q1(t, x, i) for i in missing(x))


def weight(y: tuple[int, ...], weighted: bool = True) -> F:
    if not weighted:
        return F(1)
    return F((1 + y[1]) if y[0] == y[2] else 0)


@lru_cache(None)
def partition(t: int, x: State, weighted: bool = True) -> F:
    return sum((q_complete(t, x, y) * weight(y, weighted)
                for y in product((0, 1), repeat=N)), F(0))


def clamped_partition(t: int, x: State, A: tuple[int, ...], a: tuple[int, ...],
                      weighted: bool = True) -> F:
    # Crucial: the original q_t factors at positions in A are retained.
    return sum((q_complete(t, x, y) * weight(y, weighted)
                for y in product((0, 1), repeat=N)
                if all(y[i] == v for i, v in zip(A, a))), F(0))


def rho_ref(t: int, x: State, A: tuple[int, ...]) -> F:
    M = missing(x)
    if not set(A).issubset(M) or len(A) != len(set(A)):
        return F(0)
    if not M:
        return F(int(not A))
    if t == T - 1:
        return F(int(tuple(A) == M))
    b = F(t + 1, 3)
    return b ** len(A) * (1 - b) ** (len(M) - len(A))


def rho_prop(t: int, x: State, A: tuple[int, ...], epsilon: F = F(1, 4)) -> F:
    if not 0 < epsilon <= 1:
        raise ValueError('Strict mode requires 0 < epsilon <= 1')
    M = missing(x)
    if not M or t == T - 1:
        return rho_ref(t, x, A)
    guide = (M[0],)
    # Marginal probability under the MIXTURE, not the chosen branch alone.
    return epsilon * rho_ref(t, x, A) + (1 - epsilon) * int(A == guide)


def transition(t: int, x: State, A: tuple[int, ...], a: tuple[int, ...],
               weighted: bool = True, epsilon: F = F(1, 4)):
    nxt = pin(x, A, a)
    qA = multiply(q1(t, x, i) if v else 1 - q1(t, x, i) for i, v in zip(A, a))
    K = rho_ref(t, x, A) * qA
    Z = partition(t, x, weighted)
    Zclamp = clamped_partition(t, x, A, a, weighted)
    if Z == 0:
        raise ValueError('Zero proposal mass: not automatically logical UNSAT')
    R = rho_prop(t, x, A, epsilon) * Zclamp / Z
    if R == 0:
        return nxt, K, R, None
    Znext = partition(t + 1, nxt, weighted)
    G = K * Znext / (R * Z)
    # Independent expression from v7.1 equation 22.
    G2 = (rho_ref(t, x, A) / rho_prop(t, x, A, epsilon)) * qA * Znext / Zclamp
    assert G == G2
    return nxt, K, R, G


def exact_distributions(weighted: bool = True, epsilon: F = F(1, 4)) -> dict:
    reference = defaultdict(F)
    proposal = defaultdict(F)
    corrected = defaultdict(F)
    count = 0
    Z0 = partition(0, INITIAL, weighted)

    def walk(t: int, x: State, pK: F, pR: F, pG: F) -> None:
        nonlocal count
        if t == T:
            assert not missing(x)
            y = tuple(int(v) for v in x)
            W = weight(y, weighted)
            reference[y] += pK * W
            proposal[y] += pR
            corrected[y] += Z0 * pR * pG
            assert pK * W == Z0 * pR * pG
            count += 1
            return
        sumK = F(0)
        sumR = F(0)
        for A in subsets(missing(x)):
            for a in product((0, 1), repeat=len(A)):
                nxt, K, R, G = transition(t, x, A, a, weighted, epsilon)
                sumK += K
                sumR += R
                if R:
                    walk(t + 1, nxt, pK * K, pR * R, pG * G)
                else:
                    # With positive q, Zclamp=0 excludes only terminal-zero paths.
                    assert K == 0 or partition(t + 1, nxt, weighted) == 0
        assert sumK == 1
        assert sumR == 1

    walk(0, INITIAL, F(1), F(1), F(1))
    zref = sum(reference.values(), F(0))
    zcorr = sum(corrected.values(), F(0))
    assert zref == zcorr
    assert sum(proposal.values(), F(0)) == 1
    target = {y: v / zref for y, v in reference.items()}
    corrected_norm = {y: v / zcorr for y, v in corrected.items()}
    assert target == corrected_norm
    tv = sum((abs(proposal.get(y, F(0)) - target.get(y, F(0)))
              for y in set(proposal) | set(target)), F(0)) / 2
    return dict(paths=count, Z0=Z0, target_normalizer=zref, target=target,
                direct=dict(proposal), corrected=corrected_norm, direct_tv=tv,
                corrected_tv=F(0))


def serializable_report() -> dict:
    r = exact_distributions()
    def table(d):
        return {''.join(map(str, y)): {'fraction': str(v), 'float': float(v)}
                for y, v in sorted(d.items())}
    return {
        'scope': 'newly constructed, fully enumerated 3-bit / 3-round oracle',
        'not_claimed': ['music training', 'finite-particle exactness', 'upstream reproduction'],
        'arithmetic': 'fractions.Fraction',
        'valid_proposal_paths': r['paths'],
        'initial_surrogate_Z': str(r['Z0']),
        'reference_terminal_normalizer': str(r['target_normalizer']),
        'direct_total_variation': float(r['direct_tv']),
        'weighted_enumeration_total_variation': 0.0,
        'target': table(r['target']), 'direct': table(r['direct']),
        'weighted_enumeration': table(r['corrected']),
        'assertions': ['reference kernels normalized', 'proposal kernels normalized',
                       'two incremental-weight expressions agree',
                       'pathwise telescoping identity', 'normalized target recovered'],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(serializable_report(), indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
