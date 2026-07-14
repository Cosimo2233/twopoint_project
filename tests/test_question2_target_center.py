from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

from twopoint_project.contrl import question2_target_center


class Question2TargetCenterArgsTest(unittest.TestCase):
    def test_parse_args_validates_conf_threshold_env_without_exposing_arg(self) -> None:
        with patch.dict(os.environ, {"CENTER_CONF_THRESHOLD": "0.73"}), patch.object(sys, "argv", ["prog"]):
            args = question2_target_center.parse_args()

        self.assertFalse(hasattr(args, "conf_threshold"))

    def test_parse_args_rejects_invalid_conf_threshold_env(self) -> None:
        with patch.dict(os.environ, {"CENTER_CONF_THRESHOLD": "1.73"}), patch.object(sys, "argv", ["prog"]):
            with self.assertRaises(ValueError):
                question2_target_center.parse_args()


if __name__ == "__main__":
    unittest.main()
