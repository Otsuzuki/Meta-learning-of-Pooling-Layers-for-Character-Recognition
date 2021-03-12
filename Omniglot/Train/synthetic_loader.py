"""Meta dataloader for synthetic problems."""

import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as data
from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor
from torchmeta.utils.data import BatchMetaDataLoader

class MiniImagenetDataset:
    def __init__(self, device, problem="default", task_num=16, n_way=5, imgsz=28, k_spt=1, k_qry=19):
        self.device = device
        self.task_num = task_num
        self.n_way, self.imgsz = n_way, imgsz
        self.k_spt, self.k_qry = k_spt, k_qry
        assert k_spt + k_qry <= 20, "Max 20 k_spt + k_20"
        class_augmentations = [Rotation([90, 180, 270])]
        meta_train_dataset = MiniImagenet("data",
                                      transform=Compose([Resize(self.imgsz), ToTensor()]),
                                      target_transform=Categorical(num_classes=self.n_way),
                                      num_classes_per_task=self.n_way,
                                      meta_train=True,
                                      class_augmentations=class_augmentations,
                                      download=True
                                    )
        meta_val_dataset = MiniImagenet("data",
                                      transform=Compose([Resize(self.imgsz), ToTensor()]),
                                      target_transform=Categorical(num_classes=self.n_way),
                                      num_classes_per_task=self.n_way,
                                      meta_val=True,
                                      class_augmentations=class_augmentations,
                                    )
        meta_test_dataset = MiniImagenet("data",
                                      transform=Compose([Resize(self.imgsz), ToTensor()]),
                                      target_transform=Categorical(num_classes=self.n_way),
                                      num_classes_per_task=self.n_way,
                                      meta_test=True,
                                      class_augmentations=class_augmentations,
                                    )
        self.train_dataset = ClassSplitter(meta_train_dataset, shuffle=True, num_train_per_class=k_spt, num_test_per_class=k_qry)
        self.val_dataset = ClassSplitter(meta_val_dataset, shuffle=True, num_train_per_class=k_spt, num_test_per_class=k_qry)
        self.test_dataset = ClassSplitter(meta_test_dataset, shuffle=True, num_train_per_class=k_spt, num_test_per_class=k_qry)

    def next(self, n_tasks, mode="train"):
        if mode == 'train':
                train_dataloader = BatchMetaDataLoader(self.train_dataset, batch_size=n_tasks, num_workers=0)
                dataiter = iter(train_dataloader)
                data = dataiter.next()
                x_spts, y_spts = data["train"]
                x_qrys, y_qrys = data["test"]
                data = [x_spts.to(self.device), y_spts.to(self.device), x_qrys.to(self.device), y_qrys.to(self.device)]
                return data
        if mode == 'val':
                val_dataloader = BatchMetaDataLoader(self.val_dataset, batch_size=n_tasks, num_workers=0)
                dataiter = iter(val_dataloader)
                data = dataiter.next()
                x_spts, y_spts = data["train"]
                x_qrys, y_qrys = data["test"]
                data = [x_spts.to(self.device), y_spts.to(self.device), x_qrys.to(self.device), y_qrys.to(self.device)]
                return data
        if mode == 'test':
                test_dataloader = BatchMetaDataLoader(self.test_dataset, batch_size=n_tasks, num_workers=0)
                dataiter = iter(test_dataloader)
                data = dataiter.next()
                x_spts, y_spts = data["train"]
                x_qrys, y_qrys = data["test"]
                data = [x_spts.to(self.device), y_spts.to(self.device), x_qrys.to(self.device), y_qrys.to(self.device)]
                return data

class OmniglotDataset:
    def __init__(self, device, problem="default", task_num=16, n_way=5, imgsz=28, k_spt=1, k_qry=19):
        self.device = device
        self.task_num = task_num
        self.n_way, self.imgsz = n_way, imgsz
        self.k_spt, self.k_qry = k_spt, k_qry
        assert k_spt + k_qry <= 20, "Max 20 k_spt + k_20"
        class_augmentations = [Rotation([90, 180, 270])]
        meta_train_dataset = Omniglot("data",
                                      transform=Compose([Resize(self.imgsz), ToTensor()]),
                                      target_transform=Categorical(num_classes=self.n_way),
                                      num_classes_per_task=self.n_way,
                                      meta_train=True,
                                      class_augmentations=class_augmentations,
                                      download=True
                                    )
        meta_val_dataset = Omniglot("data",
                                      transform=Compose([Resize(self.imgsz), ToTensor()]),
                                      target_transform=Categorical(num_classes=self.n_way),
                                      num_classes_per_task=self.n_way,
                                      meta_val=True,
                                      class_augmentations=class_augmentations,
                                    )
        meta_test_dataset = Omniglot("data",
                                      transform=Compose([Resize(self.imgsz), ToTensor()]),
                                      target_transform=Categorical(num_classes=self.n_way),
                                      num_classes_per_task=self.n_way,
                                      meta_test=True,
                                      class_augmentations=class_augmentations,
                                    )
        self.train_dataset = ClassSplitter(meta_train_dataset, shuffle=True, num_train_per_class=k_spt, num_test_per_class=k_qry)
        self.val_dataset = ClassSplitter(meta_val_dataset, shuffle=True, num_train_per_class=k_spt, num_test_per_class=k_qry)
        self.test_dataset = ClassSplitter(meta_test_dataset, shuffle=True, num_train_per_class=k_spt, num_test_per_class=k_qry)

    def next(self, n_tasks, mode="train"):
        if mode == 'train':
                train_dataloader = BatchMetaDataLoader(self.train_dataset, batch_size=n_tasks, num_workers=0)
                dataiter = iter(train_dataloader)
                data = dataiter.next()
                x_spts, y_spts = data["train"]
                x_qrys, y_qrys = data["test"]
                data = [x_spts.to(self.device), y_spts.to(self.device), x_qrys.to(self.device), y_qrys.to(self.device)]
                return data
        if mode == 'val':
                val_dataloader = BatchMetaDataLoader(self.val_dataset, batch_size=n_tasks, num_workers=0)
                dataiter = iter(val_dataloader)
                data = dataiter.next()
                x_spts, y_spts = data["train"]
                x_qrys, y_qrys = data["test"]
                data = [x_spts.to(self.device), y_spts.to(self.device), x_qrys.to(self.device), y_qrys.to(self.device)]
                return data
        if mode == 'test':
                test_dataloader = BatchMetaDataLoader(self.test_dataset, batch_size=n_tasks, num_workers=0)
                dataiter = iter(test_dataloader)
                data = dataiter.next()
                x_spts, y_spts = data["train"]
                x_qrys, y_qrys = data["test"]
                data = [x_spts.to(self.device), y_spts.to(self.device), x_qrys.to(self.device), y_qrys.to(self.device)]
                return data