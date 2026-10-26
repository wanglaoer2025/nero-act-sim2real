"""Catch a pointwise loss being mistaken for conditional position supervision."""
import unittest
import torch

from paired import paired_response_loss,worst_prefix_position_loss


class PairedResponseTests(unittest.TestCase):
    def test_response_is_independent_of_shared_pose_bias(self):
        target=torch.tensor([[[[-.0003,0.,0.],[-.0006,0.,0.]],
                               [[.0003,0.,0.],[.0006,0.,0.]]]],dtype=torch.float64)
        prediction=target+torch.tensor([.1,.2,.3])
        self.assertLess(float(paired_response_loss(prediction,target)),1e-20)

    def test_wrong_response_gets_opposite_endpoint_gradients(self):
        target=torch.tensor([[[[-.0003,0.,0.],[-.0006,0.,0.]],
                               [[.0003,0.,0.],[.0006,0.,0.]]]],dtype=torch.float64)
        prediction=torch.zeros_like(target,requires_grad=True)
        loss=paired_response_loss(prediction,target);loss.backward()
        self.assertGreater(float(loss),0.)
        self.assertTrue(torch.all(prediction.grad[:,0,:,0]>0))
        self.assertTrue(torch.all(prediction.grad[:,1,:,0]<0))

    def test_normalization_does_not_ignore_small_teacher_response(self):
        small=torch.zeros(1,2,2,3);small[:,1,:,0]=.0001
        large=small*10
        a=paired_response_loss(torch.zeros_like(small),small)
        b=paired_response_loss(torch.zeros_like(large),large)
        self.assertAlmostEqual(float(a),float(b),places=5)

    def test_worst_prefix_is_not_diluted_by_easier_samples(self):
        prediction=torch.zeros(2,2,3,requires_grad=True)
        target=torch.zeros_like(prediction);target[0,:,0]=.001;target[1,:,0]=.0001
        loss=worst_prefix_position_loss(prediction,target);loss.backward()
        self.assertAlmostEqual(float(loss.detach()),10.,places=5)
        self.assertGreater(float(prediction.grad[0].abs().sum()),0.)
        self.assertEqual(float(prediction.grad[1].abs().sum()),0.)


if __name__=='__main__':unittest.main()
