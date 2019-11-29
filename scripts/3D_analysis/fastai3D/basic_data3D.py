#!/usr/bin/env python3

#currently this script is not in use

from torch.utils.data import DataLoader
from fastai.basic_data import *
from fastai.torch_core import *


class DataLoader_c(DataLoader):
	
	def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, pad_idx=0,
				 num_workers=None, pin_memory=False, drop_last=False, pre_pad=True, half=False,
				 transpose=False, transpose_y=False):
		self.dataset,self.batch_size,self.num_workers = dataset,batch_size,num_workers
		self.pin_memory,self.drop_last,self.pre_pad = pin_memory,drop_last,pre_pad
		self.transpose,self.transpose_y,self.pad_idx,self.half = transpose,transpose_y,pad_idx,half

		if batch_sampler is not None:
			if batch_size != 1 or shuffle or sampler is not None or drop_last:
				raise ValueError('batch_sampler is mutually exclusive with '
								 'batch_size, shuffle, sampler, and drop_last')
			self.batch_size=batch_sampler.batch_size
			#batch_size=self.batch_size

		if sampler is not None and shuffle:
			raise ValueError('sampler is mutually exclusive with shuffle')

		if batch_sampler is None:
			if sampler is None:
				sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
			batch_sampler = BatchSampler(sampler, batch_size, drop_last)

		if num_workers is None:
			self.num_workers = num_cpus()

		self.sampler = sampler
		self.batch_sampler = batch_sampler

class  DeviceDataLoader_c(DeviceDataLoader):

	"Bind a `DataLoader_c` to a `torch.device`."
	dl: DataLoader
	device: torch.device
	tfms: List[Callable]=None
	collate_fn: Callable=data_collate

	@classmethod
	def create(cls, dataset:Dataset, bs:int=64, shuffle:bool=False, device:torch.device=defaults.device,
			   tfms:Collection[Callable]=tfms, num_workers:int=defaults.cpus, collate_fn:Callable=data_collate, **kwargs:Any):
		"Create DeviceDataLoader from `dataset` with `bs` and `shuffle`: process using `num_workers`."
		return cls(DataLoader_c(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, **kwargs),
				   device=device, tfms=tfms, collate_fn=collate_fn)


class DataBunch_c(DataBunch):

	"Bind `train_dl`,`valid_dl` and `test_dl` in a data object."

	def __init__(self, train_dl:DataLoader_c, valid_dl:DataLoader_c, fix_dl:DataLoader_c=None, test_dl:Optional[DataLoader_c]=None,
				 device:torch.device=None, dl_tfms:Optional[Collection[Callable]]=None, path:PathOrStr='.',
				 collate_fn:Callable=data_collate, no_check:bool=False):
		self.dl_tfms = listify(dl_tfms)
		self.device = defaults.device if device is None else device
		assert not isinstance(train_dl,DeviceDataLoader_c)
		def _create_dl(dl, **kwargs):
			if dl is None: return None
			return DeviceDataLoader_c(dl, self.device, self.dl_tfms, collate_fn, **kwargs)
		self.train_dl,self.valid_dl,self.fix_dl,self.test_dl = map(_create_dl, [train_dl,valid_dl,fix_dl,test_dl])
		if fix_dl is None: self.fix_dl = self.train_dl.new(shuffle=False, drop_last=False)
		self.single_dl = _create_dl(DataLoader_c(valid_dl.dataset, batch_size=1, num_workers=0))
		self.path = Path(path)
		if not no_check: self.sanity_check()

	@classmethod
	def create(cls, train_ds:Dataset, valid_ds:Dataset, test_ds:Optional[Dataset]=None, path:PathOrStr='.', bs:int=64,
			   val_bs:int=None, num_workers:int=defaults.cpus, dl_tfms:Optional[Collection[Callable]]=None,
			   device:torch.device=None, collate_fn:Callable=data_collate, no_check:bool=False, **dl_kwargs)->'DataBunch_c':
		"Create a `DataBunch` from `train_ds`, `valid_ds` and maybe `test_ds` with a batch size of `bs`. Passes `**dl_kwargs` to `DataLoader_c()`"
		datasets = cls._init_ds(train_ds, valid_ds, test_ds)
		val_bs = ifnone(val_bs, bs)
		dls = [DataLoader_c(d, b, shuffle=s, drop_last=s, num_workers=num_workers, **dl_kwargs) for d,b,s in
			   zip(datasets, (bs,val_bs,val_bs,val_bs), (True,False,False,False)) if d is not None]
		return cls(*dls, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)

	#is this needed for the custom bunch?
	def dl(self, ds_type:DatasetType=DatasetType.Valid)->DeviceDataLoader_c:
		"Returns appropriate `Dataset` for validation, training, or test (`ds_type`)."
		#TODO: refactor
		return (self.train_dl if ds_type == DatasetType.Train else
				self.test_dl if ds_type == DatasetType.Test else
				self.valid_dl if ds_type == DatasetType.Valid else
				self.single_dl if ds_type == DatasetType.Single else
				self.fix_dl)

	@property
	def dls(self)->List[DeviceDataLoader_c]:
		"Returns a list of all DeviceDataLoaders_c. If you need a specific DeviceDataLoader_c, access via the relevant property (`train_dl`, `valid_dl`, etc) as the index of DLs in this list is not guaranteed to remain constant."
		res = [self.train_dl, self.fix_dl, self.single_dl]
		# Preserve the original ordering of Train, Valid, Fix, Single, Test Data Loaders
		# (Unknown/not verified as of 1.0.47 whether there are other methods explicitly using DLs their list index)
		if self.valid_dl: res.insert(1, self.valid_dl)
		return res if not self.test_dl else res + [self.test_dl]

	def add_test(self, items:Iterator, label:Any=None, tfms=None, tfm_y=None)->None:
		"Add the `items` as a test set. Pass along `label` otherwise label them with `EmptyLabel`."
		self.label_list.add_test(items, label=label, tfms=tfms, tfm_y=tfm_y)
		vdl = self.valid_dl
		dl = DataLoader_c(self.label_list.test, vdl.batch_size, shuffle=False, drop_last=False, num_workers=vdl.num_workers)
		self.test_dl = DeviceDataLoader_c(dl, vdl.device, vdl.tfms, vdl.collate_fn)

	def _grab_dataset(self, dl:DataLoader_c):
		ds = dl.dl.dataset
		while hasattr(ds, 'dataset'): ds = ds.dataset
		return ds