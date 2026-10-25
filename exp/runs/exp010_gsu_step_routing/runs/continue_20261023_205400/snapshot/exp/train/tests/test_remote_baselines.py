"""Offline target selection checks; full-model loss/backward checks run remotely."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_remote_baselines as runner


class BaselineTargetTests(unittest.TestCase):
    def test_selects_standard_and_hybrid_language_modules(self):
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.ModuleDict({'q_proj': torch.nn.Linear(2, 2)})
        layer.linear_attn = torch.nn.ModuleDict({'in_proj_qkv': torch.nn.Linear(2, 6)})
        layer.mlp = torch.nn.ModuleDict({'up_proj': torch.nn.Linear(2, 4)})
        model.layers.append(layer)
        wrapper = torch.nn.Module()
        wrapper.model = model
        wrapper.lm_head = torch.nn.Linear(2, 10)
        targets = runner.target_modules(wrapper)
        self.assertEqual(len(targets), 3)
        self.assertTrue(any('linear_attn.in_proj_qkv' in name for name in targets))
        self.assertNotIn('lm_head', targets)

    def test_excludes_vision_layers(self):
        wrapper = torch.nn.Module()
        wrapper.visual = torch.nn.Module()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.ModuleDict({'q_proj': torch.nn.Linear(2, 2)})
        wrapper.visual.layers = torch.nn.ModuleList([layer])
        with self.assertRaises(RuntimeError):
            runner.target_modules(wrapper)


if __name__ == '__main__':
    unittest.main()
