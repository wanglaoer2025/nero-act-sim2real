"""Predetermined training positions; the original +/-15 mm tests stay held out."""
import itertools


def offsets():
    return [v for v in itertools.product([-.02, -.01, 0., .01, .02], repeat=2)
            if v != (0., 0.)]


def classify_partition(offset):
    return 'train' if tuple(offset) in offsets() else 'held_out'
