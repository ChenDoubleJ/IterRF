import unittest

from checkpoint_utils import get_alpha_beta_state


class CheckpointUtilsTests(unittest.TestCase):
    def test_reads_new_alpha_beta_key(self):
        state = object()
        self.assertIs(get_alpha_beta_state({"alpha_beta": state}), state)

    def test_reads_legacy_misspelled_key(self):
        state = object()
        self.assertIs(get_alpha_beta_state({"alpha_beat": state}), state)


if __name__ == "__main__":
    unittest.main()

