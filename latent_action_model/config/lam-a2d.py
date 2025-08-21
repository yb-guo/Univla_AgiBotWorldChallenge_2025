import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from transformers import TrainingArguments

from prismatic.vla.datasets.pretrainAe_a2d_pretrain_v6 import ActionSpacePadder
from prismatic.vla.datasets.alpha_base_cfg import BaseDatasetArguments, BaseDataTrainingArguments, BaseModelArguments

RUNNAME = os.environ.get("RUNNAME")
DEBUG_MODE = True


class Task:
    dataset_type = "a2d"


class CommonCfg:
    type = "common_cfg"
    force_image_size = 224


class ActionHead:
    type = "action_expert"
    shift = 1
    chunk = 30


alpha_set = [
    # minimal task
    # 475, #灵巧手熨衣服
    # 446, #底盘移动双机器人搬桌子
    440,
    429,  # 触觉插内存条
    424,
    422,
    421,
    414,
    410,
    398,
    392,
    390,
    389,
    388,
    385,
    384,
    380,
    378,
    377,
    376,
    375,
    374,
    372,
    368,
    367,
    366,
    365,
    # 362,  # 叠短裤9k
    361,
    360,
    359,
    358,
    357,
    356,
    354,
    352,
    327,
]
beta_set = [
    725,
    719,
    717,
    716,
    715,
    714,
    712,
    711,
    709,
    708,
    707,
    698,
    695,
    # 694,  # 触觉
    689,
    688,
    # 683,  # 出餐-双机器人协同-机器人A-1213版-AW
    # 682,  # 出餐-双机器人协同-机器人A-1213版-AW
    # 681, # 叠短裤
    # 677,  # 触觉
    # 676,  # 触觉
    # 668,  # 触觉
    # 666,  # 触觉
    664,
    660,
    # 658, # 叠短裤
    # 622, # 灵巧手打包外卖
    621,
    619,
    616,
    613,
    609,  # 挂篮区补货
    # 608, # 注释的原因: 数据方面的问题，待解决
    607,
    604,
    603,
    602,  # 挂篮区补货
    600,
    # 599,  # 叠短裤 裤腰朝左 3k
    598,
    # 597,  # 挂篮区补货
    596,
    593,
    590,
    589,
    588,
    587,
    584,
    582,
    580,
    575,
    574,
    573,
    570,
    568,
    567,
    566,
    563,
    561,
    # 559,  # 数据量为0
    558,
    556,
    555,  # 叠短裤 裤腰朝右 2.6k
    551,
    550,
    545,
    544,
    543,  # 宝洁装箱-传送带-1213版-AW 2k
    542,
    541,
    540,
    537,
    535,
    533,
    532,  # 永磁锭装箱 1.9k
    529,
    528,
    527,
    525,
    524,
    522,
    521,
    # 520,  # 叠短裤
    515,
    512,
    511,
    510,
    509,
    508,
    507,
    506,
    505,
    # 504,  # 超市零食补货-挂钩区-1213版-AW 2.4k
    503,  # 收纳玩具--桌面--1213版--AW 3.1k
    501,
    498,
    497,
    494,
    492,  # contrain gripper and dexHand
    491,
    487,
    486,
    485,
    483,
    480,
    478,
    477,
    474,
    471,
    470,
    468,
    466,
    465,
    464,
    463,
    462,
    460,
    455,
    454,
    453,
    452,
    451,
    445,
    444,
    438,  # 将笔放入笔筒里 2.4k
    434,
    433,
    431,
    428,
    425,
    373,
    369,
    363,
    351,
]
train_task_ids = alpha_set + beta_set
train_set = {}
val_set = {}
if DEBUG_MODE:
    train_task_ids = train_task_ids[0:1]
for num in train_task_ids:
    train_set[str(num)] = {
        "use_cam_list": ["head", "hand_right", "hand_left"],
        "label_file_name": f"task_{num}_train.json",
    }
    val_set[str(num)] = {
        "use_cam_list": ["head", "hand_right", "hand_left"],
        "label_file_name": f"task_{num}_val.json",
    }


@dataclass
class DatasetArguments(BaseDatasetArguments):
    meta_json_dir: Optional[str] = "/mnt/public/hexindong/SHARE_FILES/processed_json_202502131400_split"
    data_root_dir: Optional[str] = "/mnt/public/E6"
    valid_episode_txt: Optional[str] = None
    use_train_dataset_cache: str = field(default="")
    use_eval_dataset_cache: str = field(default="")
    train_frame_sample_rate: Optional[int] = field(default=None)
    eval_frame_sample_rate: Optional[int] = field(default=6)
    train_episode_sample_rate: Optional[int] = field(default=None)
    eval_episode_sample_rate: Optional[int] = field(default=None)
    sharding_by_eposide: bool = field(default=True)
    online_process_mp_cnt: int = field(default=1)
    force_image_size: int = field(default=CommonCfg.force_image_size)

    dataset_task_cfg: Optional[Dict[str, str]] = field(default_factory=lambda: train_set)
    episode_processors: Optional[List[Dict]] = field(
        default_factory=lambda: [dict(type="EpisodeProcessorLoadOnlyImage")]
    )
    dataset_processors: Optional[List[Dict]] = field(
        default_factory=lambda: [
            dict(
                type="DatasetTargetDualArmOnlyImage",
                action_chunk_size=ActionHead.chunk,
                action_shift=ActionHead.shift,
                mp_cnt=20,
            ),
        ]
    )
    runtime_processors: Optional[List[Dict]] = field(
        default_factory=lambda: [
            dict(type="RuntimeVideoPreprocessLoad"),
        ]
    )

    # Evaluation related configs
    eval_dataset_task_cfg: Optional[Dict[str, str]] = field(default_factory=lambda: val_set)
    eval_episode_processors: Optional[List[Dict]] = field(
        default_factory=lambda: [dict(type="EpisodeProcessorLoadOnlyImage")]
    )
    eval_dataset_processors: Optional[List[Dict]] = field(
        default_factory=lambda: [
            dict(
                type="DatasetTargetDualArmOnlyImage",
                action_chunk_size=ActionHead.chunk,
                action_shift=ActionHead.shift,
                mp_cnt=20,
            ),
        ]
    )
    eval_runtime_processors: Optional[List[Dict]] = field(
        default_factory=lambda: [
            dict(type="RuntimeVideoPreprocessLoad"),
        ]
    )


@dataclass
class ModelArguments(BaseModelArguments):
    model_name_or_path: str = field(default="/mnt/public/liuyi/Data/internvl/InternVL2_5-2B")
    grad_checkpoint: bool = field(default=False)
    # LAPA Config
    latent_training: bool = field(default=True)
    training_phase: int = field(default=1)
    lam_in_dim: int = 3
    lam_model_dim: int = 1024
    lam_latent_dim: int = 128
    lam_num_latents: int = 32
    lam_patch_size: int = 8
    lam_enc_blocks: int = 8
    lam_dec_blocks: int = 8
    lam_num_heads: int = 8
    lam_dropout: float = 0.0


@dataclass
class DataTrainingArguments(BaseDataTrainingArguments):
    max_dynamic_patch: int = field(default=6)
    max_seq_length: int = field(default=4096)
    action_chunk_size: int = field(default=ActionHead.chunk)
    dynamic_image_size: bool = field(default=False)
    pad2square: bool = field(default=False)
    force_image_size: int = field(default=CommonCfg.force_image_size)
    normalize_type: str = field(default="imagenet")
    conversation_type: int = field(default=0)
    dataset_type: Optional[str] = field(default=Task.dataset_type)


@dataclass
class AlphaTrainingArguments(TrainingArguments):
    output_dir: str = field(default=f"experiment/{RUNNAME}")
    overwrite_output_dir: bool = field(default=True)
    dataloader_num_workers: int = field(default=20 if not DEBUG_MODE else 1)
    bf16: bool = field(default=True)
    num_train_epochs: float = field(default=100.0)
    per_device_train_batch_size: int = field(default=64 if not DEBUG_MODE else 2)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=2e-5)  # used for training action expert from scratch and freeze VLM
    weight_decay: float = field(default=0.01)
    lr_scheduler_type: str = field(default="cosine")
    do_train: bool = field(default=True)
    group_by_length: bool = field(default=True)
    deepspeed: str = field(default="internvl_chat/zero_stage1_config.json")
    eval_on_start: bool = field(default=True)
    eval_strategy: str = field(default="no")
    eval_steps: int = field(default=2000 if not DEBUG_MODE else 10)
    include_for_metrics: List[str] = field(default_factory=lambda: ["inputs"])
    input_select_key_for_metric: str = field(default="task_ids")
    per_device_eval_batch_size: int = field(default=16 if not DEBUG_MODE else 2)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=2000)
    save_total_limit: int = field(default=100)
    logging_steps: int = field(default=10)
    report_to: str = field(default="tensorboard")
    label_names: List[str] = field(default_factory=lambda: ["action_gts"])


class ActionSpacePadderArguments(ActionSpacePadder):
    space_used_left = [
        "LEFT_ARM_JOINT_POSITIONS",
        # "LEFT_END_EFFECTOR_6D_POSE",
        "LEFT_GRIPPER_JOINT_POSITIONS",
    ]
    space_used_right = [
        "RIGHT_ARM_JOINT_POSITIONS",
        # "RIGHT_END_EFFECTOR_6D_POSE",
        "RIGHT_GRIPPER_JOINT_POSITIONS",
    ]
    space_used_body = [
        # "HEAD_JOINT_POSITIONS",
        # "WAIST_JOINT_POSITIONS",
        # "WAIST_LIFT_POSITIONS",
    ]


ActionSpacePadderArguments.get_space()
