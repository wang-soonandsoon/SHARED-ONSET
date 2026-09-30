from fractions import Fraction as F
from itertools import product
import unittest
from examples import reference_oracle as o


class OracleTests(unittest.TestCase):
    def test_joint_vs_independent(self):
        feasible = [x for x in product((0, 1), repeat=8)
                    if x[:4] == x[4:] and sum(x[:4]) == 2]
        self.assertEqual(len(feasible), 6)
        self.assertEqual(F(len(feasible), 256), F(3, 128))

    def test_feature_equality_not_token_equality(self):
        p = (F(1, 4), F(1, 2), F(1, 4))
        q = (F(1, 2), F(1, 6), F(1, 3))
        mass = sum((p[i] * q[j] for i in range(3) for j in range(3)
                    if (i > 0) == (j > 0)), F(0))
        identical = sum((p[i] * q[i] for i in range(3)), F(0))
        self.assertEqual(mass, F(1, 2))
        self.assertNotEqual(mass, identical)

    def test_appendix_a_numbers(self):
        p11, p00 = F(81, 82), F(1, 82)
        w11, w00 = F(2, 3), F(9)
        self.assertEqual(p11 * w11 / (p11 * w11 + p00 * w00), F(6, 7))

    def test_complete_q_is_normalized_with_evidence(self):
        x = (1, None, 0)
        self.assertEqual(sum((o.q_complete(1, x, y) for y in product((0, 1), repeat=3)), F(0)), 1)

    def test_clamp_retains_original_q(self):
        x = o.INITIAL
        p = o.q1(0, x, 0)
        zclamp = o.clamped_partition(0, x, (0,), (1,), False)
        self.assertEqual(zclamp, p)
        self.assertNotEqual(zclamp, o.partition(0, (1, None, None), False))

    def test_mixture_normalized_full_support(self):
        for t in (0, 1):
            As = tuple(o.subsets(o.missing(o.INITIAL)))
            self.assertEqual(sum((o.rho_prop(t, o.INITIAL, A) for A in As), F(0)), 1)
            self.assertTrue(all(o.rho_prop(t, o.INITIAL, A) > 0 for A in As))

    def test_final_round_completion(self):
        x = (None, 1, None)
        self.assertEqual(o.rho_prop(2, x, (0, 2)), 1)
        self.assertEqual(o.rho_prop(2, x, ()), 0)

    def test_identity_when_unconstrained_same_schedule(self):
        for A in o.subsets(o.missing(o.INITIAL)):
            for a in product((0, 1), repeat=len(A)):
                _, K, R, G = o.transition(0, o.INITIAL, A, a, False, F(1))
                self.assertEqual(K, R)
                self.assertEqual(G, 1)

    def test_unconstrained_different_schedule_still_has_weight(self):
        _, _, _, G = o.transition(0, o.INITIAL, (), (), False)
        self.assertEqual(G, o.rho_ref(0, o.INITIAL, ()) / o.rho_prop(0, o.INITIAL, ()))
        self.assertNotEqual(G, 1)

    def test_empty_batch_time_changes_twist(self):
        z0, z1 = o.partition(0, o.INITIAL), o.partition(1, o.INITIAL)
        self.assertNotEqual(z0, z1)
        _, _, _, G = o.transition(0, o.INITIAL, (), ())
        self.assertEqual(G, (o.rho_ref(0, o.INITIAL, ()) / o.rho_prop(0, o.INITIAL, ())) * z1 / z0)

    def test_early_completion_is_absorbing(self):
        x = (1, 1, 1)
        _, K, R, G = o.transition(0, x, (), ())
        self.assertEqual((K, R, G), (F(1), F(1), F(1)))
        self.assertEqual(o.partition(1, x), o.weight(x))

    def test_all_path_identities_and_target(self):
        r = o.exact_distributions()
        self.assertEqual(r['target'], r['corrected'])
        self.assertGreater(r['direct_tv'], 0)
        self.assertNotEqual(r['Z0'], r['target_normalizer'])

    def test_unbiased_Z_does_not_give_unbiased_inverse(self):
        self.assertEqual((F(1) + F(3)) / 2, 2)
        self.assertNotEqual((F(1) + F(1, 3)) / 2, F(1, 2))

    def test_proper_weighted_inner_sample(self):
        # Two candidate templates, proposal uniform, target unnormalized [1,3].
        rho = (F(1, 2), F(1, 2))
        gamma = (F(1), F(3))
        # f(template)=1[template==1]; exact RHS is 3.
        expectation = F(0)
        for r1, r2 in product((0, 1), repeat=2):
            w1, w2 = gamma[r1] / rho[r1], gamma[r2] / rho[r2]
            B = w1 + w2
            omega = B / 2
            Ef = (w1 * int(r1 == 1) + w2 * int(r2 == 1)) / B
            expectation += rho[r1] * rho[r2] * omega * Ef
        self.assertEqual(expectation, 3)

    def test_zero_mass_not_a_uniform_fallback(self):
        with self.assertRaises(ValueError):
            o.transition(0, (1, None, 0), (1,), (0,))

    def test_duplicate_auxiliary_paths_change_mass(self):
        original = F(3, 7)
        self.assertNotEqual(sum((original for _ in range(2)), F(0)), original)


if __name__ == '__main__':
    unittest.main()
