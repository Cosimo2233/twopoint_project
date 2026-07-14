from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

from twopoint_project.contrl import question2_target_center


class Question2TargetCenterArgsTest(unittest.TestCase):
    def test_parse_args_reads_conf_threshold_from_env(self) -> None:
        with patch.dict(os.environ, {"CENTER_CONF_THRESHOLD": "0.73"}), patch.object(sys, "argv", ["prog"]):
            args = question2_target_center.parse_args()

        self.assertEqual(args.conf_threshold, 0.73)


if __name__ == "__main__":
    unittest.main()
