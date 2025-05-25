# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np 

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv

from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from dreamwaq.vae import CENet, EstNet
from dreamwaq.utils import RunningMeanStd


from typing import Optional

class OnPolicyRunnerWAQ(OnPolicyRunner):
    """On-policy runner with WAQ (CENet) integration."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = 'cpu'
    ):
        # ─── Configs ───────────────────────────────────────────────────────
        # runner-specific config (기존 코드1의 runner 영역)
        self.cfg = train_cfg.get("runner", train_cfg)
        # algorithm/policy/VAE configs
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.vae_cfg = train_cfg.get("vae", {})
        
        self.device = device
        self.env = env  # LeggedRobot 환경
        
        # ─── Multi-GPU 설정 ───────────────────────────────────────────────
        # self._configure_multi_gpu() 내부에서 아래 필드를 설정한다고 가정:
        # self.multi_gpu_cfg, self.is_distributed, self.gpu_global_rank
        self._configure_multi_gpu()
        
        # ─── 학습 타입 결정 ────────────────────────────────────────────────
        alg_class_name = self.alg_cfg.get("class_name", self.cfg.get("algorithm_class_name"))
        if alg_class_name == "PPO":
            self.training_type = "rl"
        elif alg_class_name == "Distillation":
            self.training_type = "distillation"
        else:
            raise ValueError(f"Unknown algorithm class: {alg_class_name}")
        
        # ─── 관측치 차원 계산 ───────────────────────────────────────────────
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        
        # 특권 관측치 타입(resolve type of privileged observations)
        if self.training_type == "rl":
            self.privileged_obs_type = "critic" if "critic" in extras["observations"] else None
        else:  # distillation
            self.privileged_obs_type = "teacher" if "teacher" in extras["observations"] else None
        
        # 특권 관측치 차원(resolve dimensions)
        if self.privileged_obs_type:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs
        
        # ─── Policy(ActorCritic) 생성 ───────────────────────────────────────
        policy_class_name = self.policy_cfg.pop("class_name", self.cfg.get("policy_class_name"))
        policy_class = eval(policy_class_name)
        actor_critic = policy_class(
            num_obs,
            num_privileged_obs,
            self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)
        
        # ─── RND gated state 설정(Optional) ─────────────────────────────────
        rnd_cfg = self.alg_cfg.get("rnd_cfg")
        if rnd_cfg:
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Missing 'rnd_state' in observations for RND.")
            rnd_cfg["num_states"] = rnd_state.shape[1]
            rnd_cfg["weight"] *= self.env.unwrapped.step_dt
        
        # ─── Symmetry 설정(Optional) ────────────────────────────────────────
        symmetry_cfg = self.alg_cfg.get("symmetry_cfg")
        if symmetry_cfg:
            symmetry_cfg["_env"] = self.env
        
        # ─── Algorithm(PPO/Distillation) 생성 ───────────────────────────────
        # alg_cfg에서 class_name을 꺼내고 나머지를 넘겨줌
        self.alg_cfg.pop("class_name", None)
        alg_class = eval(alg_class_name)
        self.alg = alg_class(
            actor_critic,
            device=self.device,
            multi_gpu_cfg=self.multi_gpu_cfg,
            **self.alg_cfg
        )
        
        # ─── Runner 기본 설정 저장 ─────────────────────────────────────────
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg.get("empirical_normalization", False)
        
        if self.empirical_normalization:
            from rl_utils import EmpiricalNormalization  # 예시 import
            self.obs_normalizer = EmpiricalNormalization(
                shape=[num_obs], until=1e8
            ).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(
                shape=[num_privileged_obs], until=1e8
            ).to(self.device)
        else:
            import torch
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)
        
        # ─── 저장소(Storage) 및 VAE 초기화 ─────────────────────────────────
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )
        
        # VAE (CENet) 설정
        vae_class_name = self.vae_cfg.pop("class_name", self.cfg.get("vae_class_name"))
        vae_class = eval(vae_class_name)
        self.cenet = vae_class(device=self.device, **self.vae_cfg).to(self.device)
        env_cfg = self.env.cfg.env
        self.cenet.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [env_cfg.len_obs_history * env_cfg.num_observations],
            [env_cfg.num_estvel],
            [env_cfg.num_observations],
        )
        
        # ─── 로그 설정 ────────────────────────────────────────────────────
        # distributed 환경에서 rank≠0이면 로그 비활성화
        self.disable_logs = getattr(self, "is_distributed", False) and getattr(self, "gpu_global_rank", 0) != 0
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        
        # git 상태 추적용(선택)
        import rsl_rl
        self.git_status_repos = [rsl_rl.__file__]
        
        # ─── RMS 초기화 플레이스홀더(옵션) ──────────────────────────────────
        self.rms_dict = {}
        if self.cfg.get("obs_rms"):
            self.obs_rms = None
        if self.cfg.get("privileged_obs_rms"):
            self.privileged_obs_rms = None
        if self.cfg.get("true_vel_rms"):
            self.true_vel_rms = None
        
        # 환경 리셋
        _, _ = self.env.reset()
        

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # ─── Writer 초기화 ────────────────────────────────────────────────
        if self.log_dir is not None and self.writer is None and not getattr(self, "disable_logs", False):
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Choose 'neptune', 'wandb' or 'tensorboard'.")

        # ─── Distillation teacher 체크 ───────────────────────────────────────
        if getattr(self, "training_type", None) == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model not loaded. Load a teacher before distilling.")

        # ─── 에피소드 길이 랜덤 초기화 ───────────────────────────────────────
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length)
            )

        # ─── Train 모드 진입 ────────────────────────────────────────────────
        self.alg.actor_critic.train()
        self.cenet.train_mode()

        # ─── Book-keeping 버퍼 ─────────────────────────────────────────────
        ep_infos = []
        rew_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # RND 사용 시 내·외재 보상 버퍼
        if getattr(self.alg, "rnd", False):
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # ─── 분산 학습 파라미터 동기화 ───────────────────────────────────────
        if getattr(self, "is_distributed", False):
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()
            # ─── 초기 관측치 & 정규화 ───────────────────────────────────────
            obs = self.env.get_observations().to(self.device)

            if self.cfg.get("obs_rms"):
                if self.obs_rms is None:
                    self.obs_rms = RunningMeanStd(shape=obs.shape[1], device=self.device)
                self.obs_rms.update(obs.detach())
                obs = (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)

            privileged_obs = self.env.get_privileged_observations().to(self.device)
            if self.cfg.get("privileged_obs_rms"):
                if self.privileged_obs_rms is None:
                    self.privileged_obs_rms = RunningMeanStd(
                        shape=privileged_obs.shape[1], device=self.device
                    )
                self.privileged_obs_rms.update(privileged_obs.detach())
                privileged_obs = (
                    privileged_obs - self.privileged_obs_rms.mean
                ) / torch.sqrt(self.privileged_obs_rms.var + 1e-8)

            true_vel = self.env.get_true_vel().to(self.device)
            if self.cfg.get("true_vel_rms"):
                if self.true_vel_rms is None:
                    self.true_vel_rms = RunningMeanStd(shape=true_vel.shape[1], device=self.device)
                self.true_vel_rms.update(true_vel.detach())
                true_vel = (true_vel - self.true_vel_rms.mean) / torch.sqrt(self.true_vel_rms.var + 1e-8)

            # ─── Rollout 수집 ────────────────────────────────────────────────
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # CENet 입력 준비
                    obs_history = self.env.get_observation_history().to(self.device)
                    if self.cfg.get("obs_rms"):
                        obs_history = (
                            obs_history - self.obs_rms.mean
                        ) / torch.sqrt(self.obs_rms.var + 1e-8)
                    obs_history = obs_history.view(self.env.num_envs, -1)

                    est_next_obs, est_vel, mu, logvar, context_vec = self.cenet.before_action(
                        obs_history, true_vel
                    )

                    # AdaBoot
                    if self.cfg.get("ada_boot", False):
                        boot_prob = self.env.extras["episode"]["boot_prob"].item()
                        vel_input = est_vel if boot_prob > np.random.rand() else true_vel
                    else:
                        vel_input = est_vel

                    # actor/critic 관측치 구성
                    critic_obs = torch.cat((obs, vel_input, privileged_obs), dim=-1)
                    actor_obs = torch.cat((obs, vel_input, context_vec), dim=-1)

                    actions = self.alg.act(actor_obs, critic_obs)

                    # 환경 스텝
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
                    true_vel = self.env.get_true_vel().to(self.device)

                    # RMS 후처리
                    if self.cfg.get("obs_rms"):
                        self.obs_rms.update(obs.detach())
                        obs = (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)
                    if self.cfg.get("privileged_obs_rms"):
                        self.privileged_obs_rms.update(privileged_obs.detach())
                        privileged_obs = (
                            privileged_obs - self.privileged_obs_rms.mean
                        ) / torch.sqrt(self.privileged_obs_rms.var + 1e-8)
                    if self.cfg.get("true_vel_rms"):
                        self.true_vel_rms.update(true_vel.detach())
                        true_vel = (true_vel - self.true_vel_rms.mean) / torch.sqrt(self.true_vel_rms.var + 1e-8)

                    self.cenet.after_action(obs)

                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    # ─── 로그 버퍼 업데이트 ───────────────────────────────
                    if self.log_dir is not None and not getattr(self, "disable_logs", False):
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        if "reward_cv" in infos:
                            rew_infos.append(infos["reward_cv"])

                        # 총 보상/길이
                        if getattr(self.alg, "rnd", False):
                            extrinsic = rewards
                            intrinsic = self.alg.intrinsic_rewards
                            cur_ereward_sum += extrinsic
                            cur_ireward_sum += intrinsic
                            cur_reward_sum += extrinsic + intrinsic
                        else:
                            cur_reward_sum += rewards

                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                        if getattr(self.alg, "rnd", False):
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                collection_time = time.time() - start

                # returns 계산 (RL 전용)
                self.alg.compute_returns(critic_obs)

            # ─── 파라미터 업데이트 ────────────────────────────────────────────
            cen_losses = self.cenet.update()           # (total, vel, recon, kl)
            policy_losses = self.alg.update()          # (value, surrogate)

            learn_time = time.time() - (start + collection_time)

            # ─── 로깅 & 모델 저장 ───────────────────────────────────────────
            if self.log_dir is not None and not getattr(self, "disable_logs", False):
                self.log(locals())

                if it % self.save_interval == 0:
                    # RMS 저장
                    if self.cfg.get("obs_rms"):
                        self.rms_dict["obs_rms"] = self.obs_rms
                    if self.cfg.get("privileged_obs_rms"):
                        self.rms_dict["privileged_obs_rms"] = self.privileged_obs_rms
                    if self.cfg.get("true_vel_rms"):
                        self.rms_dict["true_vel_rms"] = self.true_vel_rms

                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"), infos=infos)

            # 첫 이터레이션 뒤 코드 상태 저장 (wandb/neptune)
            if it == start_iter and self.log_dir is not None and self.logger_type in ("wandb", "neptune"):
                from rsl_rl.utils.git_utils import store_code_state
                git_paths = store_code_state(self.log_dir, self.git_status_repos)
                for path in git_paths:
                    self.writer.save_file(path)

            ep_infos.clear()
            rew_infos.clear()

        # ─── 마지막 모델 저장 ───────────────────────────────────────────────
        self.current_learning_iteration += num_learning_iterations

        if self.log_dir is not None and not getattr(self, "disable_logs", False):
            if self.cfg.get("obs_rms"):
                self.rms_dict["obs_rms"] = self.obs_rms
            if self.cfg.get("privileged_obs_rms"):
                self.rms_dict["privileged_obs_rms"] = self.privileged_obs_rms
            if self.cfg.get("true_vel_rms"):
                self.rms_dict["true_vel_rms"] = self.true_vel_rms

            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"), infos=infos)


    def log(self, locs: dict, width: int = 80, pad: int = 35):
        import time, statistics
        # ─── 총 타임스텝 & 시간 업데이트 ────────────────────────────────────
        # 분산 학습 시 GPU world 크기만큼 스케일링
        world_size = getattr(self, "gpu_world_size", 1)
        collection_size = self.num_steps_per_env * self.env.num_envs * world_size
        self.tot_timesteps += collection_size
        iteration_time = locs["collection_time"] + locs["learn_time"]
        self.tot_time += iteration_time

        # ─── Episode 정보 로깅 ───────────────────────────────────────────
        ep_string = ""
        if locs.get("ep_infos"):
            for key in locs["ep_infos"][0]:
                # gather 텐서
                infotensor = torch.tensor([], device=self.device)
                for info in locs["ep_infos"]:
                    if key not in info:
                        continue
                    val = info[key]
                    if not isinstance(val, torch.Tensor):
                        val = torch.tensor([val], device=self.device)
                    if val.ndim == 0:
                        val = val.unsqueeze(0)
                    infotensor = torch.cat((infotensor, val.to(self.device)))
                mean_val = infotensor.mean()
                # 이름에 "/" 들어가 있으면 그대로, 아니면 "Episode/" prefix
                tag = key if "/" in key else f"Episode/{key}"
                self.writer.add_scalar(tag, mean_val, locs["it"])
                label = f"{key}:" if "/" in key else f"Mean episode {key}:"
                ep_string += f"{label:>{pad}} {mean_val:.4f}\n"

        # ─── CENet & Losses 로깅 ─────────────────────────────────────────
        # CENet 세부 항목 (코드 1)
        self.writer.add_scalar('CENet/beta',        self.cenet.beta,                                locs["it"])
        self.writer.add_scalar('CENet/lr',          self.cenet.optimizer.param_groups[0]['lr'],    locs["it"])
        self.writer.add_scalar('CENet/kl_loss',     locs['mean_kl_loss'],                          locs["it"])
        self.writer.add_scalar('CENet/recon_loss',  locs['mean_recon_loss'],                       locs["it"])
        self.writer.add_scalar('CENet/vel_loss',    locs['mean_vel_loss'],                         locs["it"])
        self.writer.add_scalar('CENet/total_loss',  locs['mean_total_loss'],                       locs["it"])
        # 일반 loss_dict 항목 (코드 2)
        for key, val in locs.get("loss_dict", {}).items():
            self.writer.add_scalar(f"Loss/{key}", val, locs["it"])
        # Learning rate & policy noise std (공통)
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate,                      locs["it"])
        mean_std = self.alg.actor_critic.std.mean() if hasattr(self.alg.actor_critic, "std") \
                else self.alg.policy.action_std.mean()
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(),                          locs["it"])

        # ─── 퍼포먼스 지표 ────────────────────────────────────────────────
        fps = int(collection_size / iteration_time)
        self.writer.add_scalar('Perf/total_fps',      fps,                                         locs["it"])
        self.writer.add_scalar('Perf/collection_time', locs["collection_time"],                    locs["it"])
        self.writer.add_scalar('Perf/learning_time',   locs["learn_time"],                         locs["it"])

        # ─── Train 보상 지표 ──────────────────────────────────────────────
        if len(locs.get("rewbuffer", [])) > 0:
            if getattr(self.alg, "rnd", False):
                # RND 보상
                self.writer.add_scalar('Rnd/mean_extrinsic_reward',
                                    statistics.mean(locs['erewbuffer']),            locs["it"])
                self.writer.add_scalar('Rnd/mean_intrinsic_reward',
                                    statistics.mean(locs['irewbuffer']),            locs["it"])
                self.writer.add_scalar('Rnd/weight',
                                    self.alg.rnd.weight,                           locs["it"])
            # 공통 train 지표
            self.writer.add_scalar('Train/mean_reward',
                                statistics.mean(locs['rewbuffer']),                locs["it"])
            self.writer.add_scalar('Train/mean_episode_length',
                                statistics.mean(locs['lenbuffer']),                locs["it"])
            if self.logger_type != "wandb":
                # wandb x축은 int만 지원
                self.writer.add_scalar('Train/mean_reward/time',
                                    statistics.mean(locs['rewbuffer']),                self.tot_time)
                self.writer.add_scalar('Train/mean_episode_length/time',
                                    statistics.mean(locs['lenbuffer']),                self.tot_time)

        # ─── 터미널 출력 문자열 구성 ────────────────────────────────────
        header = f" Learning iteration {locs['it']}/{locs['tot_iter']} "
        out = [ "#" * width,
                header.center(width),
                "",
                f"{'Computation:':>{pad}} {fps:.0f} steps/s "
                f"(collection: {locs['collection_time']:.3f}s, learning: {locs['learn_time']:.3f}s)",
                f"{'Mean noise std:':>{pad}} {mean_std.item():.2f}" ]

        # Losss 요약
        for key, val in locs.get("loss_dict", {}).items():
            out.append(f"{f'Mean {key} loss:':>{pad}} {val:.4f}")

        # RND 보상 요약
        if getattr(self.alg, "rnd", False) and len(locs.get("erewbuffer", [])) > 0:
            out += [
                f"{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}",
                f"{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}"
            ]

        # 공통 reward/episode length
        if len(locs.get("rewbuffer", [])) > 0:
            out += [
                f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}",
                f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}"
            ]

        # Episode info
        if ep_string:
            out.append(ep_string.rstrip())  # 마지막 개행 제거

        # Footer
        eta_sec = self.tot_time / (locs['it'] - locs['start_iter'] + 1) * \
                (locs['tot_iter'] - locs['it'])
        out += [
            "-" * width,
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}",
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s",
            f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}",
            f"{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta_sec))}"
        ]

        print("\n".join(out))


    def save(self, path: str, infos=None):
        # ─── 기본 저장할 항목들 ─────────────────────────────────────────
        # policy 네트워크: RL은 actor_critic, Distillation은 policy
        policy_net = getattr(self.alg, "actor_critic", None) or getattr(self.alg, "policy", None)
        saved_dict = {
            "model_state_dict": policy_net.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            "rms": self.rms_dict,
        }

        # ─── CENet 저장 (VAE 사용 시) ───────────────────────────────────
        if hasattr(self, "cenet"):
            saved_dict.update({
                "cenet_state_dict": self.cenet.state_dict(),
                "cenet_optimizer_state_dict": self.cenet.optimizer.state_dict(),
            })

        # ─── RND 저장 (사용 시) ─────────────────────────────────────────
        if getattr(self.alg, "rnd", False):
            saved_dict.update({
                "rnd_state_dict": self.alg.rnd.state_dict(),
                "rnd_optimizer_state_dict": self.alg.rnd_optimizer.state_dict(),
            })

        # ─── 경험적 정규화 저장 (사용 시) ────────────────────────────────
        if getattr(self, "empirical_normalization", False):
            saved_dict.update({
                "obs_norm_state_dict":              self.obs_normalizer.state_dict(),
                "privileged_obs_norm_state_dict":   self.privileged_obs_normalizer.state_dict(),
            })

        # ─── 파일로 저장 ────────────────────────────────────────────────
        torch.save(saved_dict, path)

        # ─── 외부 로거 업로드 (wandb/neptune) ───────────────────────────
        if getattr(self, "writer", None) and \
        getattr(self, "logger_type", None) in ["neptune", "wandb"] and \
        not getattr(self, "disable_logs", False):
            # Writer에 맞게 save_model 호출
            self.writer.save_model(path, self.current_learning_iteration)

        
    def load(self, path: str, load_optimizer: bool = True):
        # 1) 파일 로드
        loaded = torch.load(path)
        
        # 2) 네트워크 불러오기 (actor_critic 또는 policy)
        policy_net = getattr(self.alg, "actor_critic", None) or getattr(self.alg, "policy", None)
        resumed_training = policy_net.load_state_dict(loaded["model_state_dict"])
        
        # 3) CENet (VAE) 불러오기
        if hasattr(self, "cenet") and "cenet_state_dict" in loaded:
            self.cenet.load_state_dict(loaded["cenet_state_dict"])
        
        # 4) RND 모델 불러오기 (사용 시)
        if getattr(self.alg, "rnd", False) and "rnd_state_dict" in loaded:
            self.alg.rnd.load_state_dict(loaded["rnd_state_dict"])
        
        # 5) 경험적 정규화 불러오기 (사용 시)
        if getattr(self, "empirical_normalization", False) and "obs_norm_state_dict" in loaded:
            if resumed_training:
                # RL → 계속 학습 재개
                self.obs_normalizer.load_state_dict(loaded["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded["privileged_obs_norm_state_dict"])
            else:
                # Distillation 시작: teacher 용으로만 obs_normalizer 로드
                self.privileged_obs_normalizer.load_state_dict(loaded["obs_norm_state_dict"])
        
        # 6) 옵티마이저 상태 불러오기 (재개 중일 때만)
        if load_optimizer and resumed_training:
            # 알고리즘 옵티마이저
            if "optimizer_state_dict" in loaded:
                self.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
            # CENet 옵티마이저
            if hasattr(self, "cenet") and "cenet_optimizer_state_dict" in loaded:
                self.cenet.optimizer.load_state_dict(loaded["cenet_optimizer_state_dict"])
            # RND 옵티마이저
            if getattr(self.alg, "rnd", False) and "rnd_optimizer_state_dict" in loaded:
                self.alg.rnd_optimizer.load_state_dict(loaded["rnd_optimizer_state_dict"])
        
        # 7) RMS 정보 복원 (옵션)
        if "rms" in loaded:
            self.rms_dict = loaded["rms"]
        
        # 8) 이터레이션 카운터 복원
        if resumed_training and "iter" in loaded:
            self.current_learning_iteration = loaded["iter"]
        
        return loaded.get("infos", None)


    def get_inference_policy(self, device: str | None = None):
        """
        Returns a callable for inference that applies any necessary normalization
        before passing observations to the policy's act_inference method.
        """
        # ─── Switch to eval mode ───────────────────────────────────────────
        # ActorCritic or policy network
        net = getattr(self.alg, "actor_critic", None) or getattr(self.alg, "policy")
        net.eval()
        # VAE eval if present
        if hasattr(self, "cenet"):
            # assume cenet has an eval_mode() method
            self.cenet.eval_mode()

        # ─── Move to target device ────────────────────────────────────────
        if device is not None:
            net.to(device)
            if hasattr(self, "cenet"):
                self.cenet.to(device)
            if getattr(self, "empirical_normalization", False):
                self.obs_normalizer.to(device)
                self.privileged_obs_normalizer.to(device)

        # ─── Base inference function ──────────────────────────────────────
        policy_fn = net.act_inference

        # ─── Wrap with empirical normalization if enabled ────────────────
        if getattr(self, "empirical_normalization", False):
            def normalized_policy(obs):
                # apply observation normalizer before inference
                norm_obs = self.obs_normalizer(obs.to(device or self.device))
                return net.act_inference(norm_obs)
            policy_fn = normalized_policy

        return policy_fn


    def get_rms(self):
        return self.rms_info if (self.cfg["obs_rms"] or self.cfg["privileged_obs_rms"] or self.cfg["true_vel_rms"]) else None

    def get_inference_cenet(self, device=None):
        self.cenet.test_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.cenet.encoder.to(device)
        return self.cenet
    
    # """
    # New functions
    # """
    # train_mode
    # eval_mode
    # add_git_repo_to_log
    # _configure_multi_gpu
