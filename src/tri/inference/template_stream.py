"""Standard exact template conditioning with honest serial-work accounting.

The existing template implementation visits one rhythm template at a time and
sums that template's complete pitch/HOLD chain. Its previous preflight checked
``number_of_templates * local_states * token_choices`` against a *simultaneous*
factor-entry budget, although it never allocates that joint slab. Here only the
per-template live candidate count is compared with that budget. A large number
of templates still costs traversal time and may reach the real wall-clock limit.

Numerical summation, template order, chain computation, original-q evidence
semantics and sampling are inherited unchanged. In particular, exact sampling
still stores candidate template weights, and its genuine storage check is NOT
removed. Counter-plan, traversal-stack, live chain and backward-layer checks
remain active. The original backend and previously recorded evidence are intact.
"""
from tri.inference.music_backends import MusicExactInference


class StreamingTemplateMusicInference(MusicExactInference):
    """Serial exact template baseline; only the fictitious joint slab is removed.

    ``backend_name`` deliberately remains the superclass's internal ``template``
    dispatch key. Public requested/selected names and all statistics identify
    ``template_stream``; changing the dispatch key would select the automaton in
    the inherited query and sampling methods.
    """

    def __init__(self, spec, log_probs, budget=None):
        super().__init__(spec, log_probs, 'template', budget)
        self.requested_backend = 'template_stream'
        self.planning_stats = {
            'requested_backend': 'template_stream', 'selected_backend': 'template_stream',
            'internal_dispatch': 'template', 'template_enumeration': 'serial_exact_conditioning',
            'transition_budget_kind': 'per_template_live_chain',
        }

    def _new_stats(self):
        return {**super()._new_stats(), 'backend': 'template_stream',
                'transition_budget_kind': 'per_template_live_chain'}

    def _template_budget(self, plan, choices, local_sizes, *, store_weights=False):
        # Each chain is evaluated and discarded before visiting the next
        # template. Multiplying this live factor by plan.count charges memory
        # for a tensor that does not exist in this serial implementation.
        live_candidates = max((local_sizes[i] * len(tokens) for i, tokens in enumerate(choices)), default=0)
        self.budget.check_factor_entries(live_candidates, context='Serial template live chain transitions')
        # Retain the original conservative accounting for Python tuples/list
        # entries and all stored exact-template weights during sampling.
        extra = plan.count * (96 + len(plan.allowed) * 40) if store_weights else 0
        self.budget.check_workspace_bytes(self._base_bytes() + plan.bytes_estimate + extra,
                                          context='Template plan and sampling weights')
        return live_candidates
