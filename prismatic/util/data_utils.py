"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""
import re
import string
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple, Any
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output
    

@dataclass
class PaddedCollatorForActionPrediction_Geniesim:

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        
        init_pixel_values = [instance["init_pixel_values"] for instance in instances]
        goal_pixel_values = [instance["goal_pixel_values"] for instance in instances]

        hist_init_pixel_values, hist_goal_pixel_values = [], []
        with_hist = []
        for instance in instances:
            if "hist_init_pixel_values" in instance:
                hist_init_pixel_values.append(instance["hist_init_pixel_values"])
                hist_goal_pixel_values.append(instance["hist_goal_pixel_values"])
                with_hist.append(torch.tensor(True))
            else:
                with_hist.append(torch.tensor(False))     

        pixel_values = [instance["pixel_values"] for instance in instances]
        
        # For low-level policy training
        actions = [instance["actions"] for instance in instances]
        actions = torch.stack(actions, dim=0)

        proprio = [instance["proprio"] for instance in instances]
        proprio = torch.stack(proprio, dim=0)

        instructions = [instance["lang"] for instance in instances]


        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        pixel_values = torch.stack(pixel_values)
        init_pixel_values = torch.stack(init_pixel_values)
        goal_pixel_values = torch.stack(goal_pixel_values)
        hist_init_pixel_values = torch.stack(hist_init_pixel_values) if len(hist_init_pixel_values) > 0 else []
        hist_goal_pixel_values = torch.stack(hist_goal_pixel_values) if len(hist_goal_pixel_values) > 0 else []
        with_hist = torch.stack(with_hist)

        output = dict(
            pixel_values=pixel_values,
            init_pixel_values=init_pixel_values,
            goal_pixel_values=goal_pixel_values,
            hist_init_pixel_values=hist_init_pixel_values,
            hist_goal_pixel_values=hist_goal_pixel_values,
            instructions=instructions,
            with_hist=with_hist,
            # input_ids=input_ids,
            # attention_mask=attention_mask,
            # labels=labels,
            actions=actions,
            proprio=proprio
        )

        return output


@dataclass
class CollatorForLatentAction:
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        initial_pixel_values = [instance["initial_pixel_values"] for instance in instances]
        initial_pixel_values = torch.stack(initial_pixel_values)
        
        target_pixel_values = [instance["target_pixel_values"] for instance in instances]
        target_pixel_values = torch.stack(target_pixel_values)
        pixel_values = torch.stack([initial_pixel_values, target_pixel_values], dim=1)


        action = [torch.from_numpy(instance["action"]) for instance in instances]
        action = torch.stack(action)

        # removing all punctuation in task instruction
        task_instruction = [re.sub('[{}]'.format(string.punctuation),"",instance["task_instruction"]) for instance in instances]


        output = dict(
            videos=pixel_values,
            task_instruction=task_instruction,
            action=action,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names

        return output


@dataclass
class CollatorForMultiViewVideo:
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        initial_pixel_values = [instance["initial_pixel_values"] for instance in instances]
        initial_pixel_values = torch.stack(initial_pixel_values)
        
        target_pixel_values = [instance["target_pixel_values"] for instance in instances]
        target_pixel_values = torch.stack(target_pixel_values)
        pixel_values = torch.stack([initial_pixel_values, target_pixel_values], dim=1)


        initial_pixel_values_view2 = [instance["initial_pixel_values_view2"] for instance in instances]
        initial_pixel_values_view2 = torch.stack(initial_pixel_values_view2)
        
        target_pixel_values_view2 = [instance["target_pixel_values_view2"] for instance in instances]
        target_pixel_values_view2 = torch.stack(target_pixel_values_view2)
        pixel_values_view2 = torch.stack([initial_pixel_values_view2, target_pixel_values_view2], dim=1)
        


        action = [torch.from_numpy(instance["action"]) for instance in instances]
        action = torch.stack(action)

        # removing all punctuation in task instruction
        task_instruction = [re.sub('[{}]'.format(string.punctuation),"",instance["task_instruction"]) for instance in instances]


        output = dict(
            videos=pixel_values,
            videos_view2=pixel_values_view2,
            task_instruction=task_instruction,
            action=action,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names

        return output
    

@dataclass
class data_collator_lam:
    pixel_values_dtype: torch.dtype = torch.float32
    
    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        first = instances[0]
        batch = {}

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k in ("videos"):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.concat([f[k] for f in instances])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.concat(np.stack([f[k] for f in instances]))
                else:
                    batch[k] = torch.concat([f[k] for f in instances])
        
        return batch
