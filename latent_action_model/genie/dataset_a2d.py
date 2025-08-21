import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from typing import Any, Callable
from lightning import LightningDataModule
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data import get_worker_info
import importlib
from prismatic.util import set_global_seed
from prismatic.util.data_utils import data_collator_lam
from prismatic.vla.datasets.dataset_a2d import LAMStage1Dataset
from prismatic.vla.datasets.alpha_base_cfg import BaseDatasetArguments, BaseDataTrainingArguments
from latent_action_model.genie.lam_dataset import LAMConcatDataset

# Set constants for image processing
from PIL import Image, ImageFile, PngImagePlugin
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


def exists(var) -> bool:
    return var is not None


def default(var, val) -> Any:
    return var if exists(var) else val


def default_worker_init_fn(worker_id: int) -> None:
    torch.manual_seed(torch.initial_seed() + worker_id)
    worker_info = get_worker_info()

    if exists(worker_info):
        dataset = worker_info.dataset
        glob_start = dataset._start
        glob_end = dataset._end

        per_worker = int((glob_end - glob_start) / worker_info.num_workers)
        worker_id = worker_info.id

        dataset._start = glob_start + worker_id * per_worker
        dataset._end = min(dataset._start + per_worker, glob_end)
    
    
def build_datasets(
    dataset_args: BaseDatasetArguments,
    data_training_args: BaseDataTrainingArguments,
    is_train=True,
    ):
    dataset = LAMStage1Dataset(
        # base params
        label_file_dir=dataset_args.meta_json_dir,
        data_root_dir=dataset_args.data_root_dir,
        valid_episode_txt=dataset_args.valid_episode_txt,
        world_size=dist.get_world_size(),
        rank_id=dist.get_rank(),
        online_process_mp_cnt=dataset_args.online_process_mp_cnt,
        # a2d params
        is_train=is_train,
        image_size=data_training_args.force_image_size,
        pad2square=data_training_args.pad2square,
        normalize_type=data_training_args.normalize_type,
    )

    if is_train:
        dataset.generate_task_infos(
            dataset_cfg=dataset_args.dataset_task_cfg,
            task_episode_processors_cfg=dataset_args.episode_processors,
            task_dataset_processors_cfg=dataset_args.dataset_processors,
            task_runtime_processors_cfg=dataset_args.runtime_processors,
            shuffle=True,
            debug_one_episode=False,
            )
    else:
        dataset.generate_task_infos(
            dataset_cfg=dataset_args.eval_dataset_task_cfg,
            task_episode_processors_cfg=dataset_args.eval_episode_processors,
            task_dataset_processors_cfg=dataset_args.eval_dataset_processors,
            task_runtime_processors_cfg=dataset_args.eval_runtime_processors,
            shuffle=True,
            debug_one_episode=False,
            )
        
    return dataset


class LightningDataset(LightningDataModule):
    """
    Abstract LightningDataModule that represents a dataset we can train a Lightning module on.
    """

    def __init__(
            self,
            *args,
            batch_size: int = 8,
            num_workers: int = 64,
            train_shuffle: bool = True,
            val_shuffle: bool = False,
            val_batch_size: int = None,
            worker_init_fn: Callable = None,
            collate_fn: Callable = None,
            train_sampler: Callable = None,
            test_sampler: Callable = None,
            val_sampler: Callable = None
    ) -> None:
        super(LightningDataset, self).__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        val_batch_size = default(val_batch_size, batch_size)

        self.num_workers = 24    # For RLDS parallelism
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        # shuffle unspecified for iteratable datasets
        # self.train_shuffle = train_shuffle
        # self.val_shuffle = val_shuffle

        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.val_sampler = val_sampler
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn

    def train_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            # shuffle=self.train_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self) -> DataLoader:
        if isinstance(self.val_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            batch_size=self.val_batch_size,
            # shuffle=self.val_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )

    def test_dataloader(self) -> DataLoader:
        if isinstance(self.test_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            batch_size=self.val_batch_size,
            # shuffle=self.val_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )


class LightningA2D(LightningDataset):
    """
    This dataset samples video recorded using a random agent
    playing the gym environments defined in the Procgen Benchmark,
    see Cobbe et al. ICML (2020).
    """

    def __init__(
            self,
            data_root: str,
            data_mix: str,
            batch_size:int = 16,
            resolution: int = 256,
            num_frames: int = 16,
            episodic: bool = False,
            shuffle_buffer_size: int = 100_000,
            image_aug:bool = False,
            **kwargs
    ) -> None:
        super(LightningA2D, self).__init__(**kwargs)

        self.data_root_dir = data_root
        self.data_mix = data_mix

        self.batch_size = batch_size
        self.resolution = (resolution, resolution)
        self.num_frames = num_frames

        self.episodic = episodic
        self.shuffle_buffer_size = shuffle_buffer_size
        self.image_aug = image_aug

        self.num_workers = 24    # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        self.worker_init_fn = set_global_seed(42, get_worker_init_fn=True)

        self.collate_fn = data_collator_lam()

        self.save_hyperparameters()
        
        # Dataset initialization for train and eval
        self.all_train_datasets = []
        self.all_eval_datasets = []

        all_cfgs = []
        cfg_paths = ["/mnt/chenjin/AgiBot-World/latent_action_model/config/lam-a2d.py"]
        for cfg_path in cfg_paths:
            file_path = Path(cfg_path)
            sys.path.insert(0, str(file_path.parent))
            cfg = importlib.import_module(file_path.stem)
            all_cfgs.append(cfg)
            
        for data_cfg in all_cfgs:
            self.all_train_datasets.append(
                build_datasets(
                    dataset_args=data_cfg.DatasetArguments(),
                    data_training_args=data_cfg.DataTrainingArguments(),
                    is_train=True,
                )
            )
            self.all_eval_datasets.append(
                build_datasets(
                    dataset_args=data_cfg.DatasetArguments(),
                    data_training_args=data_cfg.DataTrainingArguments(),
                    is_train=False,
                )
            )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = LAMConcatDataset(self.all_train_datasets)
            self.val_dataset = LAMConcatDataset(self.all_eval_datasets)
            
        elif stage == "test":
            self.test_dataset = LAMConcatDataset(self.all_eval_datasets)
        else:
            raise ValueError(f"Invalid stage: {stage}")
