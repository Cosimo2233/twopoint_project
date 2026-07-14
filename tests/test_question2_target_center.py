from __future__ import annotations

import unittest

from twopoint_project.contrl import question2_target_center


class Question2TargetCenterEntryPointTest(unittest.TestCase):
    def test_module_no_longer_exposes_standalone_parser(self) -> None:
        self.assertFalse(hasattr(question2_target_center, "parse_args"))
        self.assertFalse(hasattr(question2_target_center, "main"))


if __name__ == "__main__":
    unittest.main()
