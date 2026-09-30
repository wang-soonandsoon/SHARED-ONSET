"""Exact MusicSpec backends with distinct, explicitly bounded state spaces.

``template`` sums an entire pitch/HOLD chain for every feasible shared-onset
template. ``automaton`` instead scans time with the sounding pitch, open onset
equalities and unfinished counters in its state. Neither drops local coupling.
``auto`` compares their request-specific structural work bounds; it is a planner,
not a new inference algorithm or a guarantee of universally optimal runtime.

All probabilities refer to the ORIGINAL supplied q times C exp(S). Temporary
evidence keeps old q weights. Full-vocabulary mass is never renormalized after
restricting the working pitch vocabulary. Observations have delta weight one.
"""

from collections import OrderedDict
from dataclasses import dataclass
from bisect import bisect_right
import math
from types import MappingProxyType

import numpy as np
from scipy.special import logsumexp

from tri.domain.compiler import compile_music
from tri.domain.music import MusicSpec, SILENCE, VOCAB_SIZE, verify_music
from tri.errors import BudgetExceeded, InvalidSpecification, VerificationError, ZeroMass
from tri.inference.exact import BatchSample, Budget, ExactInference, _Queries


BACKENDS = ("ve", "template", "automaton", "auto", "paired",
            "ve_aligned", "ve_minfill", "product_chain", "product_reuse",
            "paired_reuse", "paired_sparse", "template_stream", "product_prefix")


@dataclass(frozen=True)
class _DomainView:
    """Original-token query domains only; intentionally not a factor graph."""
    domains: object


@dataclass
class _TemplatePlan:
    allowed: tuple
    coefficients: tuple
    targets: tuple
    suffix: list
    count: int
    stored_states: int
    bytes_estimate: int


class _MusicVE:
    backend_name = "ve"
    requested_backend = "ve"

    def __init__(self, spec, log_probs, budget):
        self._engine = ExactInference(compile_music(spec, log_probs, budget=budget), budget=budget)
        self.planning_stats = {"selected_backend": "ve", "planner": "weighted_min_fill"}

    def __getattr__(self, name):
        return getattr(self._engine, name)

    @property
    def last_stats(self):
        return {"backend": "ve", **self._engine.last_stats}

    @property
    def stats(self):
        return self.last_stats


class MusicExactInference(_Queries):
    """Specialized finite-state exact inference with original-Y query names.

    ``graph.domains`` is a read-only original-token domain view, not an auxiliary
    factor graph. Query names are y0, y1, ... . At most 128 scalar partitions
    are cached. Template count is capped by Budget.max_oracle_assignments;
    live DP states and candidate transition slabs by max_factor_entries.
    """

    def __init__(self, spec: MusicSpec, log_probs, backend: str, budget: Budget | None = None):
        if not isinstance(spec, MusicSpec):
            raise InvalidSpecification("Expected MusicSpec")
        if backend not in ("template", "automaton", "auto"):
            raise InvalidSpecification("Specialized backend must be template, automaton, or auto")
        self.spec, self.budget = spec, budget or Budget()
        self.requested_backend = backend
        if np.iscomplexobj(log_probs):
            raise InvalidSpecification("Model log probabilities must be real")
        raw = np.asarray(log_probs)
        if raw.shape != (spec.length, VOCAB_SIZE):
            raise InvalidSpecification(f"Expected log probabilities [{spec.length}, {VOCAB_SIZE}]")
        self._input_bytes = raw.size * 8
        self._metadata_bytes = self._query_metadata_bytes()
        self.budget.check_workspace_bytes(2 * self._input_bytes + self._metadata_bytes, context="Music query snapshot")
        try:
            q = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidSpecification("Model log probabilities must be numeric") from exc
        for i in range(spec.length):
            if i in spec.observed:
                continue
            if np.isnan(q[i]).any() or np.isposinf(q[i]).any():
                raise InvalidSpecification(f"Unknown slot {i} has invalid model probabilities")
            total = float(logsumexp(q[i]))
            if not math.isfinite(total) or abs(total) > 2e-6:
                raise InvalidSpecification(f"Unknown slot {i} needs normalized full-vocabulary probabilities")
        self.log_probs = np.frombuffer(q.tobytes(), dtype=np.float64).reshape(q.shape)
        vocabulary = (0, 1) + tuple(p + 2 for p in spec.pitches)
        domains = {f"y{i}": (spec.observed[i],) if i in spec.observed else vocabulary for i in range(spec.length)}
        self.graph = _DomainView(MappingProxyType(domains))
        self._positions = {name: i for i, name in enumerate(domains)}
        self._cache = OrderedDict()
        self.last_stats = {}
        self._build_relations()
        self.backend_name = backend
        self.planning_stats = {"requested_backend": backend}
        if backend == "auto":
            self._select_backend()
        self.planning_stats["selected_backend"] = self.backend_name

    @property
    def stats(self):
        return dict(self.last_stats)

    def _query_metadata_bytes(self):
        return self.spec.length * (512 + len(self.spec.onset_counts) * 128)

    def _build_relations(self):
        parents = list(range(self.spec.length))
        def find(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i
        relevant = set()
        for a, b in self.spec.equal_onsets:
            if a != b:
                parents[find(b)] = find(a)
                relevant.update((a, b))
        for rule in self.spec.onset_counts:
            relevant.update(rule.positions)
        components = {}
        for i in sorted(relevant):
            components.setdefault(find(i), []).append(i)
        self._groups = tuple(tuple(g) for g in sorted(components.values(), key=lambda g: g[0]))
        self._group_at = {i: g for g, group in enumerate(self._groups) for i in group}
        self._rules = tuple(rule for rule in self.spec.onset_counts if rule.positions)
        self._rule_sets = tuple(frozenset(rule.positions) for rule in self._rules)
        self._coefficients = tuple(tuple(sum(i in positions for i in group) for positions in self._rule_sets)
                                   for group in self._groups)
        self._targets = tuple(rule.count for rule in self._rules)
        self._remaining = tuple(tuple(len(rule.positions) - bisect_right(rule.positions, i) for i in range(self.spec.length)) for rule in self._rules)

    def _base_bytes(self):
        return self._input_bytes + self._metadata_bytes

    def _choices(self, evidence):
        result = []
        for i in range(self.spec.length):
            name = f"y{i}"
            domain = (evidence[name],) if name in evidence else self.graph.domains[name]
            result.append(tuple(token for token in domain if i in self.spec.observed or np.isfinite(self.log_probs[i, token])))
        allowed = []
        for group in self._groups:
            bits = {0, 1}
            for i in group:
                bits &= {int(token >= 2) for token in result[i]}
            allowed.append(tuple(sorted(bits)))
            for i in group:
                result[i] = tuple(token for token in result[i] if int(token >= 2) in bits)
        return tuple(result), tuple(allowed)

    def _local(self, position, previous, token):
        spec = self.spec
        if token == 0:
            current, score = SILENCE, 0.0
        elif token == 1:
            if previous == SILENCE:
                return None
            current, score = previous, 0.0
        else:
            current = token - 2
            jump = 0 if previous == SILENCE else abs(current - previous)
            if spec.max_adjacent_interval is not None and jump > spec.max_adjacent_interval:
                return None
            score = -spec.motion_cost * jump
        if current != SILENCE:
            if position in spec.pitch_ranges:
                low, high = spec.pitch_ranges[position]
                if not low <= current <= high:
                    return None
            if position in spec.pitch_classes and current % 12 not in spec.pitch_classes[position]:
                return None
        if position in spec.fixed_soundings:
            anchor = spec.fixed_soundings[position]
            if current != (SILENCE if anchor is None else anchor):
                return None
        if position == spec.length - 1 and spec.enforce_end:
            if current != (SILENCE if spec.end_pitch is None else spec.end_pitch):
                return None
        return current, score

    def _q(self, position, token):
        return 0.0 if position in self.spec.observed else float(self.log_probs[position, token])

    @staticmethod
    def _weight(alpha, q, score):
        value = alpha + q + score
        if not math.isfinite(value):
            raise InvalidSpecification("Combined log weights exceed float64 range")
        return value

    def _new_stats(self):
        return {"backend": self.backend_name, "requested_backend": self.requested_backend,
                "cache_hit": False, "max_states": 1, "transition_attempts": 0,
                "peak_workspace_bytes": self._base_bytes()}

    def _state_budget(self, count, *, stored=0, width=0, extra_bytes=0, stats=None):
        self.budget.check_factor_entries(count, context="Music DP live states")
        # Conservative Python-dict/tuple/int allowance, including stored layers.
        size = self._base_bytes() + extra_bytes + (stored + count) * (256 + 48 * width)
        self.budget.check_workspace_bytes(size, context="Music DP state storage")
        if stats is not None:
            stats["max_states"] = max(stats["max_states"], count)
            stats["peak_workspace_bytes"] = max(stats["peak_workspace_bytes"], size)

    def _template_plan(self, allowed):
        targets, coefficients = self._targets, self._coefficients
        group_count, dimensions = len(allowed), len(targets)
        empty = _TemplatePlan(allowed, coefficients, targets, [], 0, 0, 0)
        if any(not options for options in allowed):
            return empty
        self.budget.check_factor_entries((group_count + 1) * max(1, dimensions), context="Template counter plan")
        prefix_min = [tuple(0 for _ in targets)]
        prefix_max = [tuple(0 for _ in targets)]
        for options, coefficient in zip(allowed, coefficients):
            prefix_min.append(tuple(a + min(options) * c for a, c in zip(prefix_min[-1], coefficient)))
            prefix_max.append(tuple(a + max(options) * c for a, c in zip(prefix_max[-1], coefficient)))
        if any(not low <= target <= high for low, high, target in zip(prefix_min[-1], prefix_max[-1], targets)):
            return empty
        suffix = [None] * (group_count + 1)
        suffix[-1] = {tuple(0 for _ in targets): 1}
        stored = 1
        cap = self.budget.max_oracle_assignments
        for g in range(group_count - 1, -1, -1):
            layer = {}
            self.budget.check_factor_entries(len(suffix[g + 1]) * len(allowed[g]), context="Template count-state transitions")
            for tail, ways in suffix[g + 1].items():
                for bit in allowed[g]:
                    total = tuple(a + bit * c for a, c in zip(tail, coefficients[g]))
                    if any(not low <= target - value <= high for low, high, target, value in zip(prefix_min[g], prefix_max[g], targets, total)):
                        continue
                    if total not in layer:
                        self._state_budget(len(layer) + 1, stored=stored, width=dimensions,
                                           extra_bytes=(group_count + 1) * max(1, dimensions) * 96)
                    layer[total] = min(cap + 1, layer.get(total, 0) + ways)
            suffix[g] = layer
            stored += len(layer)
        count = suffix[0].get(targets, 0)
        if count > cap:
            raise BudgetExceeded(f"Onset template count exceeds max_oracle_assignments={cap}; no pitch chains enumerated")
        size = stored * (256 + 48 * dimensions) + (group_count + 1) * max(1, dimensions) * 96
        return _TemplatePlan(allowed, coefficients, targets, suffix, count, stored, size)

    def _templates(self, plan):
        if not plan.count:
            return
        groups = len(plan.allowed)
        self.budget.check_workspace_bytes(self._base_bytes() + plan.bytes_estimate + (groups + 1) * min(groups + 1, plan.count + 1) * 64,
                                           context="Template traversal stack")
        stack = [(0, tuple(0 for _ in plan.targets), ())]
        while stack:
            g, counts, bits = stack.pop()
            if g == groups:
                yield bits
                continue
            for bit in reversed(plan.allowed[g]):
                updated = tuple(a + bit * c for a, c in zip(counts, plan.coefficients[g]))
                remainder = tuple(target - value for target, value in zip(plan.targets, updated))
                if remainder in plan.suffix[g + 1]:
                    stack.append((g + 1, updated, bits + (bit,)))

    def _local_state_bounds(self, choices):
        current = {SILENCE if self.spec.initial_pitch is None else self.spec.initial_pitch}
        sizes = [1]
        for i, tokens in enumerate(choices):
            self.budget.check_factor_entries(len(current) * len(tokens), context="Local-state reachability transitions")
            next_states = set()
            for previous in current:
                for token in tokens:
                    local = self._local(i, previous, token)
                    if local is not None:
                        next_states.add(local[0])
            current = next_states
            sizes.append(len(current))
        return tuple(sizes)

    def _template_budget(self, plan, choices, local_sizes, *, store_weights=False):
        slab = max((plan.count * local_sizes[i] * len(tokens) for i, tokens in enumerate(choices)), default=0)
        self.budget.check_factor_entries(slab, context="Template-conditioned transition slab (templates x states x tokens)")
        extra = plan.count * (96 + len(plan.allowed) * 40) if store_weights else 0
        self.budget.check_workspace_bytes(self._base_bytes() + plan.bytes_estimate + extra,
                                           context="Template plan and sampling weights")
        return slab

    def _chain(self, bits, choices, stats, *, save_layers=False, extra_bytes=0):
        initial = SILENCE if self.spec.initial_pitch is None else self.spec.initial_pitch
        current = {initial: 0.0}
        layers = [current] if save_layers else None
        stored = len(current) if save_layers else 0
        for i, domain in enumerate(choices):
            group = self._group_at.get(i)
            tokens = domain if group is None else tuple(token for token in domain if int(token >= 2) == bits[group])
            attempts = len(current) * len(tokens)
            self.budget.check_factor_entries(attempts, context="Template pitch/HOLD chain transitions")
            stats["transition_attempts"] += attempts
            next_states = {}
            for previous, alpha in current.items():
                for token in tokens:
                    local = self._local(i, previous, token)
                    if local is None:
                        continue
                    pitch, score = local
                    value = self._weight(alpha, self._q(i, token), score)
                    if pitch not in next_states:
                        self._state_budget(len(next_states) + 1, stored=stored + len(current),
                                           extra_bytes=extra_bytes, stats=stats)
                    next_states[pitch] = float(np.logaddexp(next_states.get(pitch, -np.inf), value))
            current = next_states
            if save_layers:
                layers.append(current)
                stored += len(current)
            if not current:
                return -math.inf, layers
        return float(logsumexp(list(current.values()))), layers

    def _template_partition(self, choices, allowed, stats):
        plan = self._template_plan(allowed)
        local_sizes = self._local_state_bounds(choices)
        stats["onset_templates"] = plan.count
        stats["template_counter_states"] = plan.stored_states
        stats["transition_slab_bound"] = self._template_budget(plan, choices, local_sizes)
        total = -math.inf
        for bits in self._templates(plan):
            value, _ = self._chain(bits, choices, stats, extra_bytes=plan.bytes_estimate)
            total = float(np.logaddexp(total, value))
        return total

    def _automaton_layout(self, allowed):
        """State layouts change with time; completed relations are forgotten."""
        unknown = tuple(g for g, members in enumerate(self._groups) if len(members) > 1 and len(allowed[g]) > 1)
        layout_bytes = (self.spec.length * 256
                        + 128 * sum(self._groups[g][-1] - self._groups[g][0] for g in unknown)
                        + 256 * sum(rule.positions[-1] - rule.positions[0] + 1 for rule in self._rules))
        self.budget.check_workspace_bytes(self._base_bytes() + layout_bytes, context="Automaton relation/counter layouts")
        layouts = []
        for i in range(self.spec.length):
            before_eq = tuple(g for g in unknown if self._groups[g][0] < i <= self._groups[g][-1])
            after_eq = tuple(g for g in unknown if self._groups[g][0] <= i < self._groups[g][-1])
            group = self._group_at.get(i)
            eq_index = before_eq.index(group) if group in before_eq else None
            after_selectors = tuple(before_eq.index(g) if g in before_eq else -1 for g in after_eq)
            before_counts = tuple(r for r, rule in enumerate(self._rules) if rule.positions[0] < i <= rule.positions[-1])
            after_counts = tuple(r for r, rule in enumerate(self._rules) if rule.positions[0] <= i < rule.positions[-1])
            live_rules = tuple(r for r, rule in enumerate(self._rules) if rule.positions[0] <= i <= rule.positions[-1])
            count_indices = {r: k for k, r in enumerate(before_counts)}
            layouts.append((before_eq, after_eq, eq_index, after_selectors,
                            before_counts, after_counts, live_rules, count_indices))
        return layouts, layout_bytes

    def _automaton_step(self, position, state, token, layout):
        previous, equal_bits, counts = state
        local = self._local(position, previous, token)
        if local is None:
            return None
        pitch, score = local
        _, _, eq_index, after_selectors, _, after_counts, live_rules, count_indices = layout
        onset = int(token >= 2)
        if eq_index is not None and equal_bits[eq_index] != onset:
            return None
        next_bits = tuple(onset if index == -1 else equal_bits[index] for index in after_selectors)
        next_counts = {}
        for r in live_rules:
            value = counts[count_indices[r]] if r in count_indices else 0
            value += onset if position in self._rule_sets[r] else 0
            target = self._rules[r].count
            if value > target or value + self._remaining[r][position] < target:
                return None
            if self._rules[r].positions[-1] == position and value != target:
                return None
            next_counts[r] = value
        return (pitch, next_bits, tuple(next_counts[r] for r in after_counts)), score

    def _automaton_bounds(self, choices, allowed, local_sizes):
        layouts, layout_bytes = self._automaton_layout(allowed)
        bounds = []
        for i, layout in enumerate(layouts):
            before_eq, _, _, _, before_counts, _, _, _ = layout
            bound = local_sizes[i] * (2 ** len(before_eq))
            for r in before_counts:
                prefix_min = prefix_max = tail_min = tail_max = 0
                for p in self._rules[r].positions:
                    bits = {int(token >= 2) for token in choices[p]}
                    if not bits:
                        bound = 0
                        break
                    if p < i:
                        prefix_min += min(bits)
                        prefix_max += max(bits)
                    else:
                        tail_min += min(bits)
                        tail_max += max(bits)
                low = max(prefix_min, self._rules[r].count - tail_max)
                high = min(prefix_max, self._rules[r].count - tail_min)
                bound *= max(0, high - low + 1)
            bounds.append(bound)
        transitions = [size * len(tokens) for size, tokens in zip(bounds, choices)]
        return {"max_state_upper_bound": max(bounds, default=1),
                "max_transition_upper_bound": max(transitions, default=0),
                "transition_work_upper_bound": sum(transitions),
                "layout_bytes": layout_bytes,
                "max_open_equalities": max((len(layout[0]) for layout in layouts), default=0),
                "max_open_counts": max((len(layout[4]) for layout in layouts), default=0)}

    def _select_backend(self):
        choices, allowed = self._choices({})
        sizes = self._local_state_bounds(choices)
        automaton = None
        try:
            automaton = self._automaton_bounds(choices, allowed, sizes)
        except BudgetExceeded as exc:
            self.planning_stats["automaton_budget_reason"] = str(exc)
        template = None
        try:
            plan = self._template_plan(allowed)
            slab = self._template_budget(plan, choices, sizes)
            template = {"onset_templates": plan.count, "max_transition_upper_bound": slab,
                        "transition_work_upper_bound": plan.count * sum(sizes[i] * len(tokens) for i, tokens in enumerate(choices))}
        except BudgetExceeded as exc:
            self.planning_stats["template_budget_reason"] = str(exc)
        # Deterministic and transparent; bounds include actual q support and
        # observations, but can overestimate reachable combined automaton states.
        if template is None and automaton is None:
            raise BudgetExceeded("Both exact music planning strategies exceed the supplied budget")
        self.backend_name = "template" if template is not None and (automaton is None or template["transition_work_upper_bound"] <= automaton["transition_work_upper_bound"]) else "automaton"
        self.planning_stats.update(template=template, automaton=automaton,
                                   decision_basis="computed conservative transition work bounds, not measured optimality")

    def _automaton_forward(self, choices, allowed, stats, *, save_layers=False):
        initial = SILENCE if self.spec.initial_pitch is None else self.spec.initial_pitch
        current = {(initial, (), ()): 0.0}
        layouts, layout_bytes = self._automaton_layout(allowed)
        stats["layout_bytes"] = layout_bytes
        layers = [current] if save_layers else None
        stored = len(current) if save_layers else 0
        state_counts = [1]
        maximum_width = max((max(len(layout[0]) + len(layout[4]), len(layout[1]) + len(layout[5])) + 1 for layout in layouts), default=1)
        for i, (tokens, layout) in enumerate(zip(choices, layouts)):
            attempts = len(current) * len(tokens)
            self.budget.check_factor_entries(attempts, context="Automaton candidate transitions at one time step")
            stats["transition_attempts"] += attempts
            next_states = {}
            width = maximum_width if save_layers else max(len(layout[0]) + len(layout[4]), len(layout[1]) + len(layout[5])) + 1
            for previous, alpha in current.items():
                for token in tokens:
                    edge = self._automaton_step(i, previous, token, layout)
                    if edge is None:
                        continue
                    state, score = edge
                    value = self._weight(alpha, self._q(i, token), score)
                    if state not in next_states:
                        self._state_budget(len(next_states) + 1, stored=stored + len(current), width=width,
                                           extra_bytes=layout_bytes, stats=stats)
                    next_states[state] = float(np.logaddexp(next_states.get(state, -np.inf), value))
            current = next_states
            state_counts.append(len(current))
            if save_layers:
                layers.append(current)
                stored += len(current)
            if not current:
                stats["states_per_layer"] = state_counts
                return -math.inf, layers, layouts
        stats["states_per_layer"] = state_counts
        return float(logsumexp(list(current.values()))), layers, layouts

    def log_partition(self, evidence=None):
        evidence = self._evidence(evidence)
        key = tuple((name, evidence[name]) for name in self.graph.domains if name in evidence)
        if key in self._cache:
            value, stats = self._cache[key]
            self._cache.move_to_end(key)
            self.last_stats = {**stats, "cache_hit": True}
            return value
        choices, allowed = self._choices(evidence)
        stats = self._new_stats()
        if any(not tokens for tokens in choices):
            result = -math.inf
        elif self.backend_name == "template":
            result = self._template_partition(choices, allowed, stats)
        else:
            result, _, _ = self._automaton_forward(choices, allowed, stats)
        self.last_stats = stats
        self._cache[key] = (result, dict(stats))
        if len(self._cache) > 128:
            self._cache.popitem(last=False)
        return result

    def marginal_log_probs(self, variable, evidence=None):
        if variable not in self.graph.domains:
            raise InvalidSpecification(f"Unknown original-token variable {variable!r}")
        evidence = self._evidence(evidence)
        base = self.log_partition(evidence)
        if not math.isfinite(base):
            raise ZeroMass("Conditioning event has zero mass; no logical UNSAT claim")
        domain = self.graph.domains[variable]
        self.budget.check_factor_entries(len(domain), context="Music token marginal")
        self.budget.check_workspace_bytes(self._base_bytes() + len(domain) * 16, context="Music token marginal")
        result = np.full(len(domain), -np.inf)
        for j, value in enumerate(domain):
            if variable in evidence and evidence[variable] != value:
                continue
            result[j] = self.log_partition({**evidence, variable: value}) - base
        return result

    def _draw(self, options, log_weights, rng):
        if not options:
            raise ZeroMass("No positive-mass sampling continuation")
        self.budget.check_factor_entries(len(options), context="Music backward sampling options")
        self.budget.check_workspace_bytes(self._base_bytes() + len(options) * 128, context="Music backward sampling options")
        weights = np.asarray(log_weights, dtype=np.float64)
        normalization = float(logsumexp(weights))
        if not math.isfinite(normalization):
            raise ZeroMass("Cannot sample a zero-mass query")
        probabilities = np.exp(weights - normalization)
        probabilities /= probabilities.sum()
        return options[int(rng.choice(len(options), p=probabilities))]

    def _sample_template(self, choices, allowed, rng, stats):
        plan = self._template_plan(allowed)
        sizes = self._local_state_bounds(choices)
        self._template_budget(plan, choices, sizes, store_weights=True)
        stats.update(onset_templates=plan.count, template_counter_states=plan.stored_states)
        templates, weights = [], []
        for bits in self._templates(plan):
            value, _ = self._chain(bits, choices, stats, extra_bytes=plan.bytes_estimate)
            if math.isfinite(value):
                templates.append(bits)
                weights.append(value)
        selected = self._draw(templates, weights, rng)
        extra = plan.bytes_estimate + plan.count * (96 + len(plan.allowed) * 40)
        _, layers = self._chain(selected, choices, stats, save_layers=True, extra_bytes=extra)
        pitch = self._draw(list(layers[-1]), list(layers[-1].values()), rng)
        result = [None] * self.spec.length
        stored = sum(len(layer) for layer in layers)
        for i in range(self.spec.length - 1, -1, -1):
            group = self._group_at.get(i)
            tokens = choices[i] if group is None else tuple(token for token in choices[i] if int(token >= 2) == selected[group])
            candidates, weights = [], []
            self.budget.check_factor_entries(len(layers[i]) * len(tokens), context="Template backward candidates")
            self.budget.check_workspace_bytes(self._base_bytes() + extra + stored * 256 + len(layers[i]) * len(tokens) * 160,
                                               context="Template backward candidates")
            for previous, alpha in layers[i].items():
                for token in tokens:
                    local = self._local(i, previous, token)
                    if local is not None and local[0] == pitch:
                        candidates.append((previous, token))
                        weights.append(self._weight(alpha, self._q(i, token), local[1]))
            pitch, result[i] = self._draw(candidates, weights, rng)
        return tuple(result)

    def _sample_automaton(self, choices, allowed, rng, stats):
        _, layers, layouts = self._automaton_forward(choices, allowed, stats, save_layers=True)
        if not layers[-1]:
            raise ZeroMass("Cannot sample a zero-mass music automaton")
        state = self._draw(list(layers[-1]), list(layers[-1].values()), rng)
        result = [None] * self.spec.length
        stored = sum(len(layer) for layer in layers)
        maximum_width = max((len(layout[0]) + len(layout[4]) + 1 for layout in layouts), default=1)
        for i in range(self.spec.length - 1, -1, -1):
            candidates, weights = [], []
            count = len(layers[i]) * len(choices[i])
            self.budget.check_factor_entries(count, context="Automaton backward candidates")
            self.budget.check_workspace_bytes(self._base_bytes() + stats["layout_bytes"] + stored * (256 + 48 * maximum_width) + count * 160,
                                               context="Automaton backward candidates")
            for previous, alpha in layers[i].items():
                for token in choices[i]:
                    edge = self._automaton_step(i, previous, token, layouts[i])
                    if edge is not None and edge[0] == state:
                        candidates.append((previous, token))
                        weights.append(self._weight(alpha, self._q(i, token), edge[1]))
            state, result[i] = self._draw(candidates, weights, rng)
        return tuple(result)

    def sample_batch(self, variables, rng: np.random.Generator, evidence=None):
        if isinstance(variables, str):
            raise InvalidSpecification("Batch variables must be a sequence, not text")
        variables = tuple(variables)
        if len(set(variables)) != len(variables) or any(name not in self.graph.domains for name in variables):
            raise InvalidSpecification("Batch variables must be distinct original-token names")
        if not isinstance(rng, np.random.Generator):
            raise InvalidSpecification("Expected numpy.random.Generator")
        evidence = self._evidence(evidence)
        base = self.log_partition(evidence)
        if not math.isfinite(base):
            raise ZeroMass("Cannot sample a zero-mass music query")
        if not variables:
            return BatchSample({}, 0.0, base)
        choices, allowed = self._choices(evidence)
        sampling_stats = self._new_stats()
        full = self._sample_template(choices, allowed, rng, sampling_stats) if self.backend_name == "template" else self._sample_automaton(choices, allowed, rng, sampling_stats)
        if not verify_music(full, self.spec).valid:
            raise VerificationError("Exact music sampler produced an invalid original sequence")
        assignment = {name: int(full[self._positions[name]]) for name in variables}
        clamp = self.log_partition({**evidence, **assignment})
        self.last_stats = {**self.last_stats, "sampling": sampling_stats}
        return BatchSample(assignment, clamp - base, clamp)


def make_music_engine(spec: MusicSpec, log_probs, backend: str = "ve", budget: Budget | None = None):
    """Construct an exact original-music inference engine.

    Auto chooses between the two specialized DPs using computed support/width
    bounds. VE remains an explicit optimized generic baseline. No fallback ever
    removes a rule, renormalizes cropped q, or substitutes an approximate query.
    """
    if backend not in BACKENDS:
        raise InvalidSpecification(f"Unknown music backend {backend!r}; choose from {BACKENDS}")
    if backend == "ve":
        return _MusicVE(spec, log_probs, budget)
    if backend == "template_stream":
        from tri.inference.template_stream import StreamingTemplateMusicInference
        return StreamingTemplateMusicInference(spec, log_probs, budget)
    if backend == "paired":
        from tri.inference.paired import PairedMusicInference
        return PairedMusicInference(spec, log_probs, budget)
    if backend in ("ve_aligned", "ve_minfill"):
        from tri.inference.ordered_ve import OrderedMusicVE
        return OrderedMusicVE(spec, log_probs, budget,
                              order="aligned" if backend == "ve_aligned" else "minfill")
    if backend == "product_chain":
        from tri.inference.product_chain import ProductChainMusicInference
        return ProductChainMusicInference(spec, log_probs, budget)
    if backend == "product_reuse":
        from tri.inference.product_cached import CachedProductChainMusicInference
        return CachedProductChainMusicInference(spec, log_probs, budget)
    if backend == "product_prefix":
        from tri.inference.product_prefix import PrefixCachedProductChainMusicInference
        return PrefixCachedProductChainMusicInference(spec, log_probs, budget)
    if backend in ("paired_reuse", "paired_sparse"):
        from tri.inference.paired_optimized import ReusedPairedInference
        return ReusedPairedInference(spec, log_probs, budget, sparse=backend == "paired_sparse")
    return MusicExactInference(spec, log_probs, backend, budget)
