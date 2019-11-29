#!/usr/bin/env python3

from torch.utils.data import Sampler, SequentialSampler
from torch._six import int_classes as _int_classes
import random

class OrderedBatchSampler(Sampler):

	"""wraps a sequential sampler to create batchsampler with in-batch ordered samples, 
	but shuffled batches. 

	Args:

	Example: 
		>>> list(OrderedBatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
		[[0, 1, 2], [3, 4, 5], [6, 7, 8]]
		>>> list(OrderedBatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
		[[0, 1, 2], [9], [3, 4, 5], [6, 7, 8]]

	"""
	def __init__(self, sampler, batch_size =3 , drop_last = False):
		if not isinstance(sampler, SequentialSampler):
			raise ValueError("sampler should be an instance of "
							"torch.utils.data.SequentialSampler, but got sampler={}"
							.format(sampler))
		if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
						batch_size <= 0:
			raise ValueError("batch_size should be a positive integer value, "
							"but got batch_size={}".format(batch_size))
		if not isinstance(drop_last, bool):
			raise ValueError("drop_last should be a boolean value, but got "
							"drop_last={}".format(drop_last))
		self.sampler = sampler
		self.batch_size = batch_size
		self.drop_last = drop_last

	def __iter__(self):
		all_batches = []
		batch = []
		for idx in self.sampler:
			batch.append(idx)
			if len(batch) == self.batch_size:
				all_batches.append(batch)
				batch = []
		if len(batch) > 0 and not self.drop_last:
			all_batches.append(batch)

		random.shuffle(all_batches)

		for b in all_batches:
			yield b

		all_batches = [] #free memory?

	def __len__(self):
		if self.drop_last:
			return len(self.sampler) // self.batch_size
		else:
			return (len(self.sampler) + self.batch_size - 1) // self.batch_size
