"""Tests for quantization"""
import numpy as np
import unittest
import shutil
import os
import yaml


def build_model():
    import torch
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 1, 1)
            self.linear = torch.nn.Linear(224 * 224, 5)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(1, -1)
            x = self.linear(x)
            return x
    return M()


class TestBasicTuningStrategy(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('saved', ignore_errors=True)
        
    def test_run_basic_one_trial_new_api(self):
        from neural_compressor.quantization import fit
        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
        from neural_compressor.data import Datasets, DATALOADERS
        import torchvision
        
        # dataset and dataloader
        dataset = Datasets("pytorch")["dummy"](((100, 3, 224, 224)))
        dataloader = DATALOADERS["pytorch"](dataset)
        model = build_model()
        
        def fake_eval(model):
            return 1

        # tuning and accuracy criterion
        conf = PostTrainingQuantConfig(tuning_criterion=TuningCriterion(max_trials=1))
        q_model = fit(model=model, conf=conf, calib_dataloader= dataloader, eval_dataloader=dataloader)
        self.assertIsNotNone(q_model)

if __name__ == "__main__":
    unittest.main()
