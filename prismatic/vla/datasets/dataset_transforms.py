import json
import logging
import os
import random
import multiprocessing
import torchvision.transforms as T
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
import h5py
import tqdm
import torch
from torchvision import transforms as tv_transforms
np.bool = np.bool_
logger = logging.getLogger(__name__)
VIS = False


def build_latent_image_transform():

    latent_transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((224, 224)),
        ]
    )
    return latent_transform


class EpisodeProcessorLoad:
    def __init__(self, gripper_source="state"):
        self.cam_path_suffix = "_color.jpg"
        self.joint_dof = 7
        self.gripper_dof = 1
        self.gripper_source = gripper_source
        assert self.gripper_source in ["state", "action"]
        logger.info(f"[EpisodeProcessorLoad]: load gripper values from '{self.gripper_source}'!!")

    def __call__(self, episode_info):
        state_path = os.path.join(episode_info["episode_dir"], "aligned_joints.h5")
        with h5py.File(state_path, "r") as fid:
            all_abs_joint = np.array(fid["state/joint/position"], dtype=np.float32)

            if "effector" in fid[f"{self.gripper_source}"]:
                all_abs_gripper = np.array(fid[f"{self.gripper_source}/effector/position"], dtype=np.float32)
            else:
                left_abs_gripper = np.array(fid[f"{self.gripper_source}/left_effector/position"], dtype=np.float32)
                right_abs_gripper = np.array(fid[f"{self.gripper_source}/right_effector/position"], dtype=np.float32)
                if len(left_abs_gripper.shape)==1:
                    left_abs_gripper = np.expand_dims(left_abs_gripper, axis=-1)
                    right_abs_gripper = np.expand_dims(right_abs_gripper, axis=-1)
                all_abs_gripper = np.concatenate((left_abs_gripper, right_abs_gripper), axis=-1)

            # all_abs_head: [N * [head-yaw, head-pitch]]
            all_abs_head = np.array(fid["state/head/position"], dtype=np.float32)

            # all_abs_waist: [N * [joint_body_pitch, joint_lift_body]]
            all_abs_waist = np.array(fid["state/waist/position"], dtype=np.float32)

        state = {}
        state["left_arm_abs_joint"] = all_abs_joint[:, : self.joint_dof]
        state["right_arm_abs_joint"] = all_abs_joint[:, self.joint_dof :]
        state["left_arm_abs_gripper"] = all_abs_gripper[:, : self.gripper_dof]
        state["right_arm_abs_gripper"] = all_abs_gripper[:, self.gripper_dof :]
        state["head_abs_joint"] = all_abs_head
        try:
            state["waist_abs_joint"] = all_abs_waist[:, 0:1]
            state["waist_abs_lift"] = all_abs_waist[:, 1:2]
        except Exception as error:
            state["waist_abs_joint"] = np.zeros_like(all_abs_joint)[:, :1]
            state["waist_abs_lift"] = np.zeros_like(all_abs_joint)[:, :1]
            logger.error(f"{episode_info['episode_dir']} has invalid waist state dim: {all_abs_waist.shape}")
        episode_info["state"] = state

        meta_path = os.path.join(episode_info["episode_dir"], "meta_info.json")
        with open(meta_path, "r") as fid:
            meta_info = json.load(fid)

        cam_cfg = {}
        camera_list = meta_info.pop("camera_list")
        camera_type = meta_info.pop("camera_type")
        camera_fps = meta_info.pop("camera_fps")
        sensor_type = meta_info.pop("sensor_type")
        for idx, cam_name in enumerate(camera_list):
            cam_cfg[cam_name] = {
                "camera_type": camera_type[idx],
                "sensor_type": sensor_type[idx],
                "camera_fps": camera_fps[idx],
                "camera_file_name": f"{cam_name}{self.cam_path_suffix}",
            }
        meta_info["cam_cfg"] = cam_cfg
        episode_info["meta_info"] = meta_info
        meta_version = meta_info["version"]
        assert (
            meta_version >= "v0.0.2"
        ), "don't use v0.0.1 data, due to the gripper value means action instead of state."

        return episode_info


class EpisodeProcessorJoint2Eef:
    def __init__(self):
        # self.conventor = A2dJoint2Eef()
        pass

    def __call__(self, episode_info):
        state = episode_info["state"]
        all_left_eef = []
        all_right_eef = []
        try:
            pass
            # for f_idx in range(state["left_arm_abs_joint"].shape[0]):
            #     left_eef, right_eef = self.conventor.get_eef_pos(
            #         waist_pitch=state["waist_abs_joint"][f_idx][0],
            #         waist_lift=state["waist_abs_lift"][f_idx][0],
            #         left_joints=state["left_arm_abs_joint"][f_idx],
            #         right_joints=state["right_arm_abs_joint"][f_idx],
            #         head_joints=state["head_abs_joint"][f_idx],
            #     )
            #     all_left_eef.append(left_eef)
            #     all_right_eef.append(right_eef)
            # left_effector_abs_pose = np.stack(all_left_eef)
            # right_effector_abs_pose = np.stack(all_right_eef)

            # state["left_effector_abs_pose"] = left_effector_abs_pose
            # state["right_effector_abs_pose"] = right_effector_abs_pose
        except Exception as error:
            logger.error(f"{episode_info['episode_dir']} error info: {error}")
            raise RuntimeError(f"{error}")

        return episode_info


class EpisodeProcessorRelableStaticFrames:
    """Remove static frames at the start and end of the episode according to the joint values."""

    def __init__(self, threshold: float = np.pi / 180 / 30):
        self.threshold = threshold

    def __call__(self, episode_info):
        arm_abs_joint: np.ndarray = np.c_[
            episode_info["state"]["left_arm_abs_joint"], episode_info["state"]["right_arm_abs_joint"]
        ]
        arm_abs_joint_sum = arm_abs_joint.sum(axis=1)
        arm_abs_joint_diff = np.abs(np.diff(arm_abs_joint_sum))

        start_idx, end_idx = 0, len(arm_abs_joint_diff) - 1

        for i in range(len(arm_abs_joint_diff) - 1):
            if arm_abs_joint_diff[i] >= self.threshold:
                start_idx = i
                break

        for j in range(len(arm_abs_joint_diff) - 1, 0, -1):
            if arm_abs_joint_diff[j] >= self.threshold:
                end_idx = j
                break

        end_idx += 1
        if start_idx > end_idx:
            raise ValueError(
                f"[EpisodeProcessorRelableStaticFrames] episode {episode_info['spisode_id']} has no dynamic frames!"
            )
        if episode_info["label_info"]["action_config"][0]["start_frame"] < start_idx:
            episode_info["label_info"]["action_config"][0]["start_frame"] = start_idx
        if episode_info["label_info"]["action_config"][-1]["end_frame"] > end_idx:
            episode_info["label_info"]["action_config"][-1]["end_frame"] = end_idx
        return episode_info


class EpisodeProcessorInterpolateGripperValue:
    def __init__(self, downsample_ratio=15, kind="linear", g_max=120):
        self.source_key = ["right_arm_abs_gripper", "left_arm_abs_gripper"]
        self.downsample_ratio = downsample_ratio
        self.kind = kind
        self.g_max = g_max

    def interpolate_a2d_gripper(self, single_data, downsample_ratio=20, kind="linear"):
        """
        Parameters:
            single_data: numpy array, [T, k], **单个**episode的**原始**夹爪值，值域[0, 120], k可以为1或2
            downsample_ratio: int, 默认为15，降采样倍数，值越大，台阶越平滑
        Return：
            single_data_interpolated: numpy array, [T, k]
        """
        from scipy.interpolate import interp1d

        downsample_data = single_data[::downsample_ratio, :]

        # 定义插值函数
        x = np.arange(downsample_data.shape[0])
        f = interp1d(x, downsample_data, kind=kind, axis=0)

        # 生成新的插值点
        x_new = np.linspace(0, downsample_data.shape[0] - 1, num=single_data.shape[0])
        single_data_interpolated = f(x_new)

        # 限制夹爪范围，不超过120
        single_data_interpolated = np.clip(single_data_interpolated, single_data.min(), self.g_max)
        return single_data_interpolated

    def __call__(self, episode_info):
        for key in self.source_key:
            if key in episode_info["state"].keys():
                grippers = episode_info["state"][key]
                gripper_state = self.interpolate_a2d_gripper(grippers, self.downsample_ratio, self.kind)
                episode_info["state"][key] = gripper_state

        return episode_info


class EpisodeProcessorNormalizeGripperValue:
    def __init__(self, g_min: int = 0, g_max: int = 120, thresh: int = 3, verbose: bool = False):
        self.source_key = ["right_arm_abs_gripper", "left_arm_abs_gripper"]
        self.g_min = int(g_min)
        self.g_max = int(g_max)
        if self.g_min > self.g_max:
            self.g_min, self.g_max = self.g_max, self.g_min
        self.thresh = thresh
        self.verbose = verbose

    def normlize(self, gripper):
        return (np.clip(gripper, a_max=self.g_max, a_min=self.g_min) - self.g_min) / (self.g_max - self.g_min + 1e-5)

    def __call__(self, episode_info):
        for key in self.source_key:
            if key in episode_info["state"].keys():
                grippers = episode_info["state"][key]
                grippers_min = grippers.min()
                grippers_max = grippers.max()
                if self.verbose and (
                    round(grippers_min) < self.g_min - self.thresh or round(grippers_max) > self.g_max + self.thresh
                ):
                    logger.warning(
                        f"Gripper min/max value of episode {episode_info['episode_id']} exceeds {self.g_min}/{self.g_max} by margin {self.thresh}!"
                    )
                gripper_state = self.normlize(grippers)

                episode_info["state"][key] = gripper_state

        return episode_info


class EpisodeProcessorLoadOnlyImage:
    def __init__(self):
        self.cam_path_suffix = "_color.jpg"
        self.joint_dof = 7
        self.gripper_dof = 1

    def __call__(self, episode_info):

        meta_path = os.path.join(episode_info["episode_dir"], "meta_info.json")
        with open(meta_path, "r") as fid:
            meta_info = json.load(fid)

        cam_cfg = {}
        camera_list = meta_info.pop("camera_list")
        camera_type = meta_info.pop("camera_type")
        camera_fps = meta_info.pop("camera_fps")
        sensor_type = meta_info.pop("sensor_type")
        for idx, cam_name in enumerate(camera_list):
            """
            Due to the cloud update of the camera name in the camera_list of meta_info.json, this
            modification was made to be compatible with both 'head' and 'head_color' fields.
            """
            if cam_name == "head_color":
                cam_name = "head"
            cam_cfg[cam_name] = {
                "camera_type": camera_type[idx],
                "sensor_type": sensor_type[idx],
                "camera_fps": camera_fps[idx],
                "camera_file_name": f"{cam_name}{self.cam_path_suffix}",
            }
        meta_info["cam_cfg"] = cam_cfg
        episode_info["meta_info"] = meta_info
        meta_version = meta_info["version"]
        # GCENT upload v0.0.1 version data
        # assert (
        #     meta_version >= "v0.0.2"
        # ), "don't use v0.0.1 data, due to the gripper value means action instead of state."
        return episode_info


class DatasetTargetDualArmChunk:
    def __init__(
        self,
        action_chunk_size=30,
        action_shift=1,
        action_use_delta=False,
        delta_type: str = "chunk",
        gripper_use_delta=False,
        mp_cnt=1,
    ):
        self.action_chunk_size = action_chunk_size
        self.action_shift = action_shift
        self.mp_cnt = mp_cnt
        self.action_use_delta = action_use_delta
        self.delta_type = delta_type
        if delta_type is not None and delta_type not in ("chunk", "frame"):
            raise ValueError(f"[DatasetTargetDualArmChunk] Invalid `delta_type` given: {delta_type}")
        self.end_effector_use_delta = gripper_use_delta

    @staticmethod
    def normalize_angles(radius):
        radius_normed = np.mod(radius, 2 * np.pi) - 2 * np.pi * (np.mod(radius, 2 * np.pi) > np.pi)
        return radius_normed

    def get_target_action(self, state, current_idx, target_idx, tag_name, use_delta):
        if use_delta:
            target_action = state[tag_name][target_idx] - state[tag_name][current_idx]
        else:
            target_action = state[tag_name][target_idx]

        if tag_name in ["left_arm_abs_joint", "right_arm_abs_joint"]:  # TODO(hxd)
            # print(f"before: {target_action}")
            target_action = self.normalize_angles(target_action)
            # print(f"after: {target_action}")

        if tag_name in ["left_effector_abs_pose", "right_effector_abs_pose"]:
            # target_action[3:] = (target_action[3:] + np.pi) % (np.pi * 2) - np.pi
            target_action[3:] = self.normalize_angles(target_action[3:])

        return target_action

    def _get_last_frame_idx(self, action_config):
        max_frame_idx = 0
        for step in action_config:
            max_frame_idx = max(max_frame_idx, step["end_frame"])
        return max_frame_idx

    def mp_worker(self, info):
        results = []
        state = info["state"]
        try:
            meta_info = info["meta_info"]
            task_specific_cfg = info["task_specific_cfg"]
            action_config = info["label_info"]["action_config"]
            last_frame_idx = self._get_last_frame_idx(action_config)

            all_action_desc = []
            for act_step in action_config:
                all_action_desc.append(act_step["english_action_text"])
            detailed_job_description = ";".join(all_action_desc)

            for act_step in action_config:
                start_idx = act_step["start_frame"]
                stop_idx = act_step["end_frame"]
                job_description = info["english_task_name"]
                sub_job_description = act_step["english_action_text"]

                end_idx = min(stop_idx, last_frame_idx - self.action_chunk_size * self.action_shift)
                for current_idx in range(start_idx, end_idx):
                    left_arm_joint_chunk = []
                    right_arm_joint_chunk = []
                    # left_effector_pose_chunk = []
                    # right_effector_pose_chunk = []
                    left_arm_gripper_chunk = []
                    right_arm_gripper_chunk = []

                    if current_idx == end_idx-1:
                        window_size = self.action_chunk_size
                    else:
                        window_size = np.random.randint(self.action_chunk_size, self.action_chunk_size + 2)

                    for i in range(self.action_chunk_size):
                        shift_idx = current_idx + self.action_shift * (i + 1)
                        base_idx = current_idx if self.delta_type == "chunk" else shift_idx - self.action_shift

                        # self.action_use_delta = True

                        left_arm_joint_chunk.append(
                            self.get_target_action(
                                state, base_idx, shift_idx, "left_arm_abs_joint", self.action_use_delta
                            )
                        )
                        right_arm_joint_chunk.append(
                            self.get_target_action(
                                state, base_idx, shift_idx, "right_arm_abs_joint", self.action_use_delta
                            )
                        )
                        # left_effector_pose_chunk.append(
                        #     self.get_target_action(
                        #         state, base_idx, shift_idx, "left_effector_abs_pose", self.action_use_delta
                        #     )
                        # )
                        # right_effector_pose_chunk.append(
                        #     self.get_target_action(
                        #         state, base_idx, shift_idx, "right_effector_abs_pose", self.action_use_delta
                        #     )
                        # )
                        left_arm_gripper_chunk.append(
                            self.get_target_action(
                                state, base_idx, shift_idx, "left_arm_abs_gripper", self.end_effector_use_delta
                            )
                        )
                        right_arm_gripper_chunk.append(
                            self.get_target_action(
                                state, base_idx, shift_idx, "right_arm_abs_gripper", self.end_effector_use_delta
                            )
                        )

                    action_target = {
                        "left_arm_joint_positions": np.array(left_arm_joint_chunk),
                        "right_arm_joint_positions": np.array(right_arm_joint_chunk),
                        # "left_end_effector_6d_pose": np.array(left_effector_pose_chunk),
                        # "right_end_effector_6d_pose": np.array(right_effector_pose_chunk),
                        "left_gripper_joint_positions": np.array(left_arm_gripper_chunk),
                        "right_gripper_joint_positions": np.array(right_arm_gripper_chunk),
                    }
                    agent_state = {
                        "left_arm_joint_positions": state["left_arm_abs_joint"][current_idx : current_idx + 1],
                        "right_arm_joint_positions": state["right_arm_abs_joint"][current_idx : current_idx + 1],
                        # "left_end_effector_6d_pose": state["left_effector_abs_pose"][current_idx : current_idx + 1],
                        # "right_end_effector_6d_pose": state["right_effector_abs_pose"][current_idx : current_idx + 1],
                        # "head_joint_positions": state["head_abs_joint"][current_idx : current_idx + 1],
                        # "waist_joint_positions": state["waist_abs_joint"][current_idx : current_idx + 1],
                        # "waist_lift_positions": state["waist_abs_lift"][current_idx : current_idx + 1],
                        "left_gripper_joint_positions": state["left_arm_abs_gripper"][current_idx : current_idx + 1],
                        "right_gripper_joint_positions": state["right_arm_abs_gripper"][current_idx : current_idx + 1],
                    }

                    used_cam_cfg = {}
                    for cam_name in task_specific_cfg["use_cam_list"]:
                        used_cam_cfg[cam_name] = meta_info["cam_cfg"][cam_name]

                    model_target = {
                        "task_id": f"{info['task_id']}",
                        "job_id": f"{info['job_id']}",
                        "sn_code": f"{info['sn_code']}",
                        "episode_id": f"{info['episode_id']}",
                        "episode_dir": info["episode_dir"],
                        "frame_idx": f"{current_idx}",
                        "used_cam_cfg": used_cam_cfg,
                        "job_description": job_description,
                        "sub_job_description": sub_job_description,
                        "detailed_job_description": detailed_job_description,
                        "action_target": action_target,
                        "agent_state": agent_state,
                        "ee_type": meta_info["ee_type"] if "ee_type" in meta_info else None,
                        "ee_list": meta_info["ee_list"] if "ee_list" in meta_info else None,
                        "window_size": window_size,
                    }

                    results.append(model_target)
        except Exception as error:
            logger.error(f'{info["episode_dir"]} met error: {error}')
            results = None
        return results

    def __call__(self, inputs):
        dataset = inputs["dataset"]
        reformat_data = []
        if self.mp_cnt <= 1:
            for ep_info in tqdm.tqdm(dataset, desc="target_generate", mininterval=60):
                model_targets = self.mp_worker(ep_info)
                if model_targets is not None:
                    reformat_data.extend(model_targets)
        else:
            logger.info(f"[DatasetTargetDualArmChunk] use multiprocess={self.mp_cnt}")
            with multiprocessing.Pool(processes=self.mp_cnt) as pool:
                results_tmp = pool.map(self.mp_worker, dataset)
                reformat_data = []
                for item in results_tmp:
                    if item is not None:
                        reformat_data.extend(item)

        inputs["iter_dataset"] = reformat_data
        return inputs


class DatasetTargetDualArmOnlyImage:
    def __init__(
        self,
        action_chunk_size=30,
        action_shift=1,
        mp_cnt=1,
        random_len=(15, 45),
        # clip_len=1,
    ):
        self.action_chunk_size = action_chunk_size
        self.action_shift = action_shift
        self.mp_cnt = mp_cnt
        # self.clip_len = clip_len
        self.random_video_len = random_len

    def _get_last_frame_idx(self, action_config):
        max_frame_idx = 0
        for step in action_config:
            max_frame_idx = max(max_frame_idx, step["end_frame"])
        return max_frame_idx

    def mp_worker(self, info):
        results = []
        try:
            meta_info = info["meta_info"]
            task_specific_cfg = info["task_specific_cfg"]
            action_config = info["label_info"]["action_config"]
            step_num = len(action_config)
            last_frame_idx = self._get_last_frame_idx(action_config)

            for act_step in action_config:
                start_idx = act_step["start_frame"]
                stop_idx = act_step["end_frame"]
                job_description = info["english_task_name"]
                sub_job_description = act_step["english_action_text"]
                vid_len = stop_idx - start_idx

                if vid_len < self.random_video_len[1]:
                    continue

                random_length = random.randint(self.random_video_len[0], self.random_video_len[1])
                # for j in range(0, vid_len-self.random_video_len[0], clip_len):
                for j in range(0, vid_len - self.random_video_len[0]):
                    start = j
                    end = j + random_length
                    if end > vid_len:
                        end = vid_len
                        start = max(vid_len - random_length, 0)

                    used_cam_cfg = {}
                    for cam_name in task_specific_cfg["use_cam_list"]:
                        used_cam_cfg[cam_name] = meta_info["cam_cfg"][cam_name]

                    model_target = {
                        "task_id": f"{info['task_id']}",
                        "job_id": f"{info['job_id']}",
                        "sn_code": f"{info['sn_code']}",
                        "episode_id": f"{info['episode_id']}",
                        "episode_dir": info["episode_dir"],
                        "frame_idx": f"{start}",
                        "target_idx": f"{end}",
                        "used_cam_cfg": used_cam_cfg,
                        "job_description": job_description,
                        "sub_job_description": sub_job_description,
                        "random_video_len": random_length,
                        "step_num": step_num,
                    }

                    results.append(model_target)
        except Exception as error:
            logger.error(f'{info["episode_dir"]} met error: {error}')
            results = {}
        return results

    def __call__(self, inputs):
        dataset = inputs["dataset"]
        reformat_data = []
        if self.mp_cnt <= 1:
            for ep_info in tqdm.tqdm(dataset, desc="target_generate", mininterval=60):
                model_targets = self.mp_worker(ep_info)
                reformat_data.extend(model_targets)
        else:
            logger.info(f"[DatasetTargetDualArmChunk] use multiprocess={self.mp_cnt}")
            with multiprocessing.Pool(processes=self.mp_cnt) as pool:
                results_tmp = pool.map(self.mp_worker, dataset)
                reformat_data = []
                for item in results_tmp:
                    reformat_data.extend(item)
        inputs["iter_dataset"] = reformat_data
        return inputs


class RuntimePromptGeneration:
    def __init__(self, prompt_mode_list=[0]):
        self.prompt_mode_list = prompt_mode_list

    def __call__(self, inputs):
        prompt_mode = random.choice(self.prompt_mode_list)
        if prompt_mode == 0:
            prompt = f"What action should the robot take to {inputs['job_description']}?"
        elif prompt_mode == 1:
            prompt = f"The robot is performing the step of {inputs['sub_job_description']}."
        elif prompt_mode == 2:
            prompt = f"What action should the robot take to {inputs['job_description']}? The robot is performing the step of {inputs['sub_job_description']}."
        elif prompt_mode == 3:
            prompt = f"The robot is performing the step of {inputs['detailed_job_description']}?"
        elif prompt_mode == 4:
            prompt = f"What action should the robot take to {inputs['job_description']}? Place the items in the material box in front of items with the same appearance on the shelves."
        else:
            raise IndexError(f"invalid prompt_mode: {prompt_mode}")
        inputs["final_prompt"] = prompt
        return inputs


class RuntimeImagePreprocessLoad:
    def __init__(self):
        # image shape: [C,H,W]
        self.target_key = "cam_tensor_"

    def __call__(self, inputs):

        window_size = inputs["window_size"]

        if window_size == 30:

            for cam_name, item in inputs["used_cam_cfg"].items():
                cam_file_path = os.path.join(inputs["episode_dir"], "camera", inputs["frame_idx"], item["camera_file_name"])
                img = Image.open(cam_file_path).convert("RGB")
                name = item["camera_file_name"].split(".")[0]
                inputs["init_" + self.target_key + name] = img

            for cam_name, item in inputs["used_cam_cfg"].items():
                cam_file_path = os.path.join(inputs["episode_dir"], "camera", str(int(inputs["frame_idx"])+window_size-1), item["camera_file_name"])
                img = Image.open(cam_file_path).convert("RGB")
                name = item["camera_file_name"].split(".")[0]
                inputs["goal_" + self.target_key + name] = img

        elif window_size == 31:

            for cam_name, item in inputs["used_cam_cfg"].items():
                cam_file_path = os.path.join(inputs["episode_dir"], "camera", str(int(inputs["frame_idx"])+1), item["camera_file_name"])
                img = Image.open(cam_file_path).convert("RGB")
                name = item["camera_file_name"].split(".")[0]
                inputs["init_" + self.target_key + name] = img

            for cam_name, item in inputs["used_cam_cfg"].items():
                cam_file_path = os.path.join(inputs["episode_dir"], "camera", str(int(inputs["frame_idx"])+window_size-1), item["camera_file_name"])
                img = Image.open(cam_file_path).convert("RGB")
                name = item["camera_file_name"].split(".")[0]
                inputs["goal_" + self.target_key + name] = img

            for cam_name, item in inputs["used_cam_cfg"].items():
                cam_file_path = os.path.join(inputs["episode_dir"], "camera", inputs["frame_idx"], item["camera_file_name"])
                img = Image.open(cam_file_path).convert("RGB")
                name = item["camera_file_name"].split(".")[0]
                inputs["hist_init_" + self.target_key + name] = img

            for cam_name, item in inputs["used_cam_cfg"].items():
                cam_file_path = os.path.join(inputs["episode_dir"], "camera", str(int(inputs["frame_idx"])+window_size-2), item["camera_file_name"])
                img = Image.open(cam_file_path).convert("RGB")
                name = item["camera_file_name"].split(".")[0]
                inputs["hist_goal_" + self.target_key + name] = img

        return inputs


class RuntimeImageResize:
    def __init__(self, size=(224, 224)):
        # image shape: [C,H,W]
        self.source_key = "cam_tensor_"
        self.size = size

    def __call__(self, inputs):
        keys = list(inputs.keys())
        for key in keys:
            if self.source_key in key:
                img = inputs[key].resize(self.size)
                inputs[key] = img

        return inputs


class RuntimeActionNorm:
    def __init__(self, norm_keys=None, params=None, max_value=1, min_value=-1):
        self.norm_keys = norm_keys
        self.max = max_value
        self.min = min_value
        self.params = np.array(params, dtype=np.float32)

    def __call__(self, inputs):
        for key, value in inputs["action_target"].items():
            if key in self.norm_keys:
                inputs["action_target"][key] = np.clip(value * self.params, self.min, self.max)

        return inputs


class RuntimeImageAugColorJitter:
    def __init__(self, prob_to_process=0.5, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03):
        self.prob_to_process = prob_to_process
        self.source_key = "cam_tensor_"
        self.color_jitter = tv_transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, inputs):
        keys = list(inputs.keys())
        for key in keys:
            if self.source_key in key:
                if random.random() < self.prob_to_process:
                    img = self.color_jitter(inputs[key])
                    inputs[key] = img

        return inputs


class RuntimeImageAugCorrupt:
    def __init__(self, prob_to_process=0.5):
        self.prob_to_process = prob_to_process
        self.source_key = "cam_tensor_"
        # Define our sequence of augmentation steps that will be applied to every image.
        self.seq = iaa.Sequential(
            [
                # Execute one of the following noise augmentations
                iaa.OneOf(
                    [
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                        iaa.AdditiveLaplaceNoise(scale=(0.0, 0.05 * 255), per_channel=0.5),
                        iaa.AdditivePoissonNoise(lam=(0.0, 0.05 * 255), per_channel=0.5),
                    ]
                ),
                # Execute one or none of the following blur augmentations
                iaa.SomeOf(
                    (0, 1),
                    [
                        iaa.OneOf(
                            [
                                # iaa.GaussianBlur((0, 3.0)),
                                iaa.AverageBlur(k=(2, 7)),
                                # iaa.MedianBlur(k=(3, 11)),
                            ]
                        ),
                        # iaa.MotionBlur(k=(5, 7)),
                    ],
                ),
            ],
            # do all of the above augmentations in random order
            random_order=True,
        )

    def __call__(self, inputs):
        keys = list(inputs.keys())
        for key in keys:
            if self.source_key in key:
                if random.random() < self.prob_to_process:
                    image_arr = self.seq(images=np.array(inputs[key]))
                    inputs[key] = Image.fromarray(image_arr)

        return inputs


class RuntimeImageAugRandomDropImage:
    def __init__(
        self,
        prob_to_process=[0.1, 0.1, 0.1],
        images=["cam_tensor_head_color", "cam_tensor_hand_right_color", "cam_tensor_hand_left_color"],
    ):
        self.prob_to_process = prob_to_process
        self.images = images
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.image_mean = IMAGENET_MEAN

    def drop_image(self, img):
        background_color = np.array([int(x * 255) for x in self.image_mean], dtype=np.uint8).reshape(1, 1, 3)
        background_image = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * background_color
        return background_image

    def __call__(self, inputs):
        for i, input_image in enumerate(self.images):
            if random.random() < self.prob_to_process[i]:
                img = np.array(inputs[input_image])

                background_image = self.drop_image(img)
                inputs[input_image] = Image.fromarray(background_image)
        return inputs


class RuntimeVideoPreprocessLoad:
    def __init__(self):
        # image shape: [C,H,W]
        self.target_key = "cam_tensor_"
        self.main_key = ["head_color"]

        self.dataset = []
        self.video_folder = []
                
                
    def __call__(self, inputs):
        for cam_name, item in inputs["used_cam_cfg"].items():
            name = item["camera_file_name"].split(".")[0]
            if name in self.main_key:
                cam_file_path = os.path.join(
                    inputs["episode_dir"], "camera", inputs["frame_idx"], item["camera_file_name"]
                )
                img = Image.open(cam_file_path).convert("RGB")
                cam_file_path = os.path.join(
                    inputs["episode_dir"], "camera", inputs["target_idx"], item["camera_file_name"]
                )
                img_k = Image.open(cam_file_path).convert("RGB")
                initial_pixel_values = build_latent_image_transform()(img)
                target_pixel_values = build_latent_image_transform()(img_k)
                initial_pixel_values = torch.from_numpy(
                    np.array(initial_pixel_values).astype(np.float32) / 255.0
                ).permute(2, 0, 1)
                target_pixel_values = torch.from_numpy(
                    np.array(target_pixel_values).astype(np.float32) / 255.0
                ).permute(2, 0, 1)
                video = torch.stack([initial_pixel_values, target_pixel_values], dim=0).unsqueeze(0)
                inputs["videos"] = video

        return inputs


class PipelineComposer:
    def __init__(self, cfg):
        self.processor_list = []
        for processor_cfg_ in cfg:
            processor_cfg = processor_cfg_.copy()
            p_type = processor_cfg.pop("type")
            processor = eval(p_type)(**processor_cfg)
            self.processor_list.append(processor)

    def __call__(self, inputs):
        for processor in self.processor_list:
            inputs = processor(inputs)
        return inputs
