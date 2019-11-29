#!/usr/bin/env python3
"""
This custom init fuction overrides the pytorch DataLoader init so that dataloader takes the batch_size from 
bacth_sampler in case a custom sampler is used.
"""
import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import IterableDataset, Sampler, SequentialSampler, RandomSampler, BatchSampler
from torch.utils.data.dataloader import _DatasetKind

def my__init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
				 batch_sampler=None, num_workers=0, collate_fn=None,
				 pin_memory=False, drop_last=False, timeout=0,
				 worker_init_fn=None, multiprocessing_context=None):
		torch._C._log_api_usage_once("python.data_loader")

		if num_workers < 0:
			raise ValueError('num_workers option should be non-negative; '
							 'use num_workers=0 to disable multiprocessing.')

		if timeout < 0:
			raise ValueError('timeout option should be non-negative')

		self.dataset = dataset
		self.num_workers = num_workers
		self.pin_memory = pin_memory
		self.timeout = timeout
		self.worker_init_fn = worker_init_fn
		self.multiprocessing_context = multiprocessing_context

		# Arg-check dataset related before checking samplers because we want to
		# tell users that iterable-style datasets are incompatible with custom
		# samplers first, so that they don't learn that this combo doesn't work
		# after spending time fixing the custom sampler errors.
		if isinstance(dataset, IterableDataset):
			self._dataset_kind = _DatasetKind.Iterable
			# NOTE [ Custom Samplers and `IterableDataset` ]
			#
			# `IterableDataset` does not support custom `batch_sampler` or
			# `sampler` since the key is irrelevant (unless we support
			# generator-style dataset one day...).
			#
			# For `sampler`, we always create a dummy sampler. This is an
			# infinite sampler even when the dataset may have an implemented
			# finite `__len__` because in multi-process data loading, naive
			# settings will return duplicated data (which may be desired), and
			# thus using a sampler with length matching that of dataset will
			# cause data lost (you may have duplicates of the first couple
			# batches, but never see anything afterwards). Therefore,
			# `Iterabledataset` always uses an infinite sampler, an instance of
			# `_InfiniteConstantSampler` defined above.
			#
			# A custom `batch_sampler` essentially only controls the batch size.
			# However, it is unclear how useful it would be since an iterable-style
			# dataset can handle that within itself. Moreover, it is pointless
			# in multi-process data loading as the assignment order of batches
			# to workers is an implementation detail so users can not control
			# how to batchify each worker's iterable. Thus, we disable this
			# option. If this turns out to be useful in future, we can re-enable
			# this, and support custom samplers that specify the assignments to
			# specific workers.
			if shuffle is not False:
				raise ValueError(
					"DataLoader with IterableDataset: expected unspecified "
					"shuffle option, but got shuffle={}".format(shuffle))
			elif sampler is not None:
				# See NOTE [ Custom Samplers and IterableDataset ]
				raise ValueError(
					"DataLoader with IterableDataset: expected unspecified "
					"sampler option, but got sampler={}".format(sampler))
			elif batch_sampler is not None:
				# See NOTE [ Custom Samplers and IterableDataset ]
				raise ValueError(
					"DataLoader with IterableDataset: expected unspecified "
					"batch_sampler option, but got batch_sampler={}".format(batch_sampler))
		else:
			self._dataset_kind = _DatasetKind.Map

		if sampler is not None and shuffle:
			raise ValueError('sampler option is mutually exclusive with '
							 'shuffle')

		if batch_sampler is not None:
			# auto_collation with custom batch_sampler
			if batch_size != 1 or shuffle or sampler is not None or drop_last:
				raise ValueError('batch_sampler option is mutually exclusive '
								 'with batch_size, shuffle, sampler, and '
								 'drop_last')
			batch_size = batch_sampler.batch_size
			drop_last = batch_sampler.drop_last
		elif batch_size is None:
			# no auto_collation
			if shuffle or drop_last:
				raise ValueError('batch_size=None option disables auto-batching '
								 'and is mutually exclusive with '
								 'shuffle, and drop_last')

		if sampler is None:  # give default samplers
			if self._dataset_kind == _DatasetKind.Iterable:
				# See NOTE [ Custom Samplers and IterableDataset ]
				sampler = _InfiniteConstantSampler()
			else:  # map-style
				if shuffle:
					sampler = RandomSampler(dataset)
				else:
					sampler = SequentialSampler(dataset)

		if batch_size is not None and batch_sampler is None:
			# auto_collation without custom batch_sampler
			batch_sampler = BatchSampler(sampler, batch_size, drop_last)

		self.batch_size = batch_size
		self.drop_last = drop_last
		self.sampler = sampler
		self.batch_sampler = batch_sampler

		if collate_fn is None:
			if self._auto_collation:
				collate_fn = _utils.collate.default_collate
			else:
				collate_fn = _utils.collate.default_convert

		self.collate_fn = collate_fn
		self.__initialized = True

old_dl_init = my__init__

#from fastai
def intercept_args(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
				 num_workers=0, collate_fn=default_collate, pin_memory=True, drop_last=False,
				 timeout=0, worker_init_fn=None):
	self.init_kwargs = {'batch_size':batch_size, 'shuffle':shuffle, 'sampler':sampler, 'batch_sampler':batch_sampler,
						'num_workers':num_workers, 'collate_fn':collate_fn, 'pin_memory':pin_memory,
						'drop_last': drop_last, 'timeout':timeout, 'worker_init_fn':worker_init_fn}
	old_dl_init(self, dataset, **self.init_kwargs)

torch.utils.data.DataLoader.__init__ = intercept_args