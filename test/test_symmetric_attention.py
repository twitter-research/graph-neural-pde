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

  def test_sym_row_max(self):
    dense = [[1,0,9,0],
             [2,0,8,0],
             [0,3,0,7],
             [0,5,4,6]]
    retval_max = torch.tensor([10, 10, 10, 15])
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    edge = tensor([[0, 1, 2, 3, 3, 3, 2, 1, 0], [0, 0, 1, 2, 1, 3, 3, 2, 2]])
    vals, max = sym_row_max(edge, x, 4)
    self.assertTrue(max == 15)
    self.assertTrue(torch.all(torch.eq(vals, x/15)))

    # add a self-loops
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0.5, 9.25])
    edge = tensor([[0, 1, 2, 3, 3, 3, 2, 1, 0, 1, 2], [0, 0, 1, 2, 1, 3, 3, 2, 2, 1, 2]])
    retval_max = torch.tensor([10, 10.5, 19.25, 15])
    vals, max = sym_row_max(edge, x, 4)
    self.assertTrue(max == 19.25)
    self.assertTrue(torch.all(torch.eq(vals, x/19.25)))
