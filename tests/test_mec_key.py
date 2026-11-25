"""Keys of Markov equivalence classes used for deduplicating candidates."""

import unittest

from cscd.mixed_graph import ARROW, TAIL, empty_marks, set_edge
from cscd.pipeline_v3 import mec_key, mec_key_pag


def chain(names, edges):
    """Marks with directed edges a -> b."""
    marks = empty_marks(names)
    for a, b in edges:
        set_edge(marks, a, b, TAIL, ARROW)
    return marks


class TestMecKeys(unittest.TestCase):
    def test_equivalent_dags_share_both_keys(self):
        names = ["a", "b", "c"]
        g1 = chain(names, [("a", "b"), ("b", "c")])
        g2 = chain(names, [("c", "b"), ("b", "a")])
        self.assertEqual(mec_key(g1, names), mec_key(g2, names))
        self.assertEqual(mec_key_pag(g1, names), mec_key_pag(g2, names))

    def test_collider_differs_under_both_keys(self):
        names = ["a", "b", "c"]
        g1 = chain(names, [("a", "b"), ("b", "c")])
        g2 = chain(names, [("a", "b"), ("c", "b")])
        self.assertNotEqual(mec_key(g1, names), mec_key(g2, names))
        self.assertNotEqual(mec_key_pag(g1, names), mec_key_pag(g2, names))

    def test_discriminating_path_collider(self):
        """<x, w, b, y> discriminates b. The MAGs differ only in whether b is a
        collider, so they are not Markov equivalent. They have the same skeleton
        and the same unshielded collider (at w), so the skeleton-and-collider key
        merges them, and the PAG key must not."""
        names = ["x", "w", "b", "y"]
        common = [("x", "w"), ("w", "y")]
        m1 = chain(names, common)
        set_edge(m1, "w", "b", ARROW, ARROW)
        set_edge(m1, "b", "y", ARROW, ARROW)   # b <-> y: b is a collider on the path
        m2 = chain(names, common)
        set_edge(m2, "w", "b", ARROW, ARROW)
        set_edge(m2, "b", "y", TAIL, ARROW)    # b -> y: b is not a collider
        self.assertEqual(mec_key(m1, names), mec_key(m2, names))
        self.assertNotEqual(mec_key_pag(m1, names), mec_key_pag(m2, names))


if __name__ == "__main__":
    unittest.main()
