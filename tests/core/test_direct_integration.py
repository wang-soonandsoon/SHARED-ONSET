"""Independent original-token checks across compiler, decoder and path math."""

from itertools import combinations, product
import math
import unittest

import numpy as np

from tri.domain.music import HOLD, REST, MusicSpec, note_token, verify_music
from tri.sampling.direct import direct_decode
from tri.sampling.schedules import RevealSchedule
from tri.sampling.weights import log_path_increment


VOCABULARY = (REST, HOLD, note_token(60), note_token(64))


def context_probabilities(state, noise):
    """Deliberately time/state inconsistent; no exact terminal assumption."""
    probabilities = np.full((len(state), 130), 0.0003, dtype=np.float64)
    visible = sum(1 for value in state if value is not None)
    for i in range(len(state)):
        probabilities[i, list(VOCABULARY)] = [
            0.2 + 0.3 * noise, 0.05 + 0.03 * visible,
            0.6 - 0.2 * noise + 0.1 * i,
            0.2 + 0.08 * visible + 0.25 * noise,
        ]
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.log(probabilities)


def original_masses(spec, state, log_q):
    """Enumerate original Y only; never construct factors or auxiliaries."""
    choices = [(value,) if value is not None else VOCABULARY for value in state]
    masses = {}
    for values in product(*choices):
        checked = verify_music(values, spec)
        if checked.valid:
            log_weight = checked.soft_score + sum(log_q[i, token] for i, token in enumerate(values) if state[i] is None)
            masses[values] = math.exp(log_weight)
    return masses


def schedules(missing, step, steps, epsilon):
    if not missing or step == steps - 1:
        yield missing, 1.0, 1.0
        return
    rate = 1.0 / (steps - step)
    for size in range(len(missing) + 1):
        for selected in combinations(missing, size):
            reference = rate ** size * (1 - rate) ** (len(missing) - size)
            proposal = epsilon * reference + (1 - epsilon) * (selected == missing[:1])
            yield selected, reference, proposal


class DirectIntegrationTests(unittest.TestCase):
    def test_actual_decoder_trace_matches_original_y_sums(self):
        spec = MusicSpec(3, (60, 64), initial_pitch=60, observed={1: HOLD},
                         equal_onsets=((0, 2),), motion_cost=0.21)
        saw_empty, saw_joint = False, False
        for seed in range(20):
            epsilon = 1.0 if seed == 0 else 0.35
            result = direct_decode(spec, context_probabilities, steps=3, seed=seed,
                                   epsilon=epsilon, track_path_weights=True)
            state = tuple(spec.observed.get(i) for i in range(spec.length))
            for row in result.trace:
                q = context_probabilities(state, row["noise"])
                masses = original_masses(spec, state, q)
                z = sum(masses.values())
                selected, values = row["positions"], row["values"]
                clamp = sum(weight for y, weight in masses.items() if all(y[i] == value for i, value in zip(selected, values)))
                self.assertAlmostEqual(row["log_z"], math.log(z), places=11)
                self.assertAlmostEqual(row["log_z_clamped"], math.log(clamp), places=11)
                self.assertAlmostEqual(row["log_batch_probability"], math.log(clamp / z), places=11)
                self.assertAlmostEqual(row["log_q_batch"], sum(q[i, value] for i, value in zip(selected, values)), places=11)
                missing = tuple(i for i, value in enumerate(state) if value is None)
                reference, proposal = next((r, p) for batch, r, p in schedules(missing, row["step"], 3, epsilon) if batch == selected)
                self.assertAlmostEqual(row["log_rho_reference"], math.log(reference), places=11)
                self.assertAlmostEqual(row["log_rho_proposal"], math.log(proposal), places=11)
                updated = list(state)
                for i, value in zip(selected, values):
                    updated[i] = value
                state = tuple(updated)
                if all(value is not None for value in state):
                    next_z = math.exp(verify_music(state, spec).soft_score)
                else:
                    next_q = context_probabilities(state, 1 - (row["step"] + 1) / 3)
                    next_z = sum(original_masses(spec, state, next_q).values())
                expected_g = math.log(reference / proposal) + row["log_q_batch"] + math.log(next_z / clamp)
                self.assertAlmostEqual(row["log_z_next"], math.log(next_z), places=11)
                self.assertAlmostEqual(row["log_g"], expected_g, places=11)
                saw_empty |= len(selected) == 0
                saw_joint |= len(selected) > 1
            self.assertEqual(result.tokens, state)
            relative_weight = sum(row["log_g"] for row in result.trace)
            self.assertAlmostEqual(result.log_relative_path_weight, relative_weight, places=11)
            self.assertAlmostEqual(result.initial_log_partition, result.trace[0]["log_z"], places=11)
            self.assertAlmostEqual(result.log_path_weight, result.initial_log_partition + relative_weight, places=11)
            self.assertTrue(verify_music(result.tokens, spec).valid)
        self.assertTrue(saw_empty)
        self.assertTrue(saw_joint)

    def test_exhaustive_music_paths_recover_reference_times_terminal_weight(self):
        spec = MusicSpec(2, (60, 64), initial_pitch=60,
                         equal_onsets=((0, 1),), motion_cost=0.23)
        steps, epsilon = 2, 0.4
        schedule = RevealSchedule(steps, epsilon)
        initial = (None, None)
        z0 = sum(original_masses(spec, initial, context_probabilities(initial, 1.0)).values())
        direct, corrected, target = {}, {}, {}

        def visit(state, step, reference_probability, proposal_probability, log_weight):
            if step == steps or all(value is not None for value in state):
                terminal_weight = math.exp(verify_music(state, spec).soft_score)
                direct[state] = direct.get(state, 0.0) + proposal_probability
                corrected[state] = corrected.get(state, 0.0) + z0 * proposal_probability * math.exp(log_weight)
                target[state] = target.get(state, 0.0) + reference_probability * terminal_weight
                return
            log_q = context_probabilities(state, 1 - step / steps)
            masses = original_masses(spec, state, log_q)
            z = sum(masses.values())
            missing = tuple(i for i, value in enumerate(state) if value is None)
            for selected, rho_ref, rho_prop in schedules(missing, step, steps, epsilon):
                actual_ref, actual_prop = schedule.log_probs(step, missing, selected)
                self.assertAlmostEqual(actual_ref, math.log(rho_ref), places=12)
                self.assertAlmostEqual(actual_prop, math.log(rho_prop), places=12)
                possible_values = {tuple(y[i] for i in selected) for y in masses}
                for values in possible_values:
                    clamp = sum(mass for y, mass in masses.items() if all(y[i] == value for i, value in zip(selected, values)))
                    updated = list(state)
                    for i, value in zip(selected, values):
                        updated[i] = value
                    updated = tuple(updated)
                    log_q_batch = sum(log_q[i, value] for i, value in zip(selected, values))
                    if all(value is not None for value in updated):
                        next_z = math.exp(verify_music(updated, spec).soft_score)
                    else:
                        next_q = context_probabilities(updated, 1 - (step + 1) / steps)
                        next_z = sum(original_masses(spec, updated, next_q).values())
                    log_g = log_path_increment(log_rho_reference=actual_ref, log_rho_proposal=actual_prop,
                                               log_q_batch=log_q_batch, log_z_next=math.log(next_z),
                                               log_z_clamped=math.log(clamp))
                    visit(updated, step + 1,
                          reference_probability * rho_ref * math.exp(log_q_batch),
                          proposal_probability * rho_prop * clamp / z,
                          log_weight + log_g)

        visit(initial, 0, 1.0, 1.0, 0.0)
        self.assertAlmostEqual(sum(direct.values()), 1.0, places=12)
        self.assertEqual(set(corrected), set(target))
        for y in target:
            self.assertAlmostEqual(corrected[y], target[y], places=12)
        target_z = sum(target.values())
        direct_tv = .5 * sum(abs(direct[y] - target[y] / target_z) for y in target)
        self.assertGreater(direct_tv, 0.001)


if __name__ == "__main__":
    unittest.main()
