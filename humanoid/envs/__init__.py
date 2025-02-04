# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
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
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .pai.pai_config import PaiCfg, PaiCfgPPO
from .pai.pai_env import PaiFreeEnv

from .pai_cl.pai_cl_config import Pai_Cl_Cfg,Pai_Cl_CfgPPO
from .pai_cl.pai_cl_env import Pai_Cl_FreeEnv

from .pai_cl_0905.pai_cl_0905_config import Pai_Cl_0905_Cfg,Pai_Cl_0905_CfgPPO
from .pai_cl_0905.pai_cl_0905_env import Pai_Cl_0905_FreeEnv

from .hi_cl_0925.hi_cl_0905_config import Hi_Cl_0925_Cfg,Hi_Cl_0925_CfgPPO
from .hi_cl_0925.hi_cl_0905_env import Hi_Cl_0925_FreeEnv
  
from humanoid.utils.task_registry import task_registry
task_registry.register( "pai_ppo", PaiFreeEnv, PaiCfg(), PaiCfgPPO() )
task_registry.register( "pai_cl_ppo", Pai_Cl_FreeEnv, Pai_Cl_Cfg(), Pai_Cl_CfgPPO() )
task_registry.register( "pai_cl_0905_ppo", Pai_Cl_0905_FreeEnv, Pai_Cl_0905_Cfg(), Pai_Cl_0905_CfgPPO() )
task_registry.register( "hi_cl_0925_ppo", Hi_Cl_0925_FreeEnv, Hi_Cl_0925_Cfg(), Hi_Cl_0925_CfgPPO() )
