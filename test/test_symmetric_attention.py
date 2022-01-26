#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test attention
"""
import unittest
import torch
from torch import tensor
from utils import make_symmetric, sym_row_max, make_symmetric_unordered


class SymmetricAttentionTests(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self) -> None:
    pass

  def test_make_symmetric_unordered(self):
    x = torch.tensor([1, 2, 3, 4])
    edge = tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    retval = torch.tensor([1.5, 1.5, 3.5, 3.5])
    vals = make_symmetric_unordered(edge, x)
    self.assertTrue(torch.all(torch.eq(vals, retval)))
    # add a self-loop
    x = torch.tensor([1, 2, 3, 4, 5])
    edge = tensor([[0, 1, 1, 2, 0], [1, 0, 2, 1, 0]])
    retval = torch.tensor([1.5, 1.5, 3.5, 3.5, 5])
    vals = make_symmetric_unordered(edge, x)
    self.assertTrue(torch.all(torch.eq(vals, retval)))
