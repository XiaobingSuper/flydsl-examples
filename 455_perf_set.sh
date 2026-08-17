# To point to rocm build
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
export HIP_DEVICE_LIB_PATH=$ROCM_PATH/lib/llvm/amdgcn/bitcode

# Extra recommended config
export HSA_ENABLE_SDMA=1
export HSA_USE_SVM=1
export HSA_XNACK=1

sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

curl -sSL http://dcgpuval-storage.amd.com/users/harnsing_pharaoh/dram_settings/set_dram_settings.py | sudo python3

curl -sSL http://dcgpuval-storage.amd.com/users/muku/MI450x_ScaleUp_PerfScripts/MI450_disTxIdle_may13.py | sudo python3
curl -sSL http://dcgpuval-storage.amd.com/users/kstraube/set_tdc_limits_mi450.py | sudo python3
curl -sSL http://dcgpuval-storage.amd.com/users/jelui/mi45x_scripts/disable_gcea_link_mgr.py | sudo python3
curl -sSL http://dcgpuval-storage.amd.com/users/jelui/mi45x_scripts/set_cp_hpd_enable_offload_check.py | sudo python3

curl -sSL http://dcgpuval-storage.amd.com/users/harnsing_pharaoh/kll_optimization/kll_optimization_mi450.py | sudo python3

# Below script includes fix from 50us debug
curl -fsSL http://dcgpuval-storage.amd.com/users/tifyeung/Perf/Perf_DisSdpDisc_MGCG.sh | sudo bash

curl -fsSL http://dcgpuval-storage.amd.com/users/jelui/mi45x_scripts/mgcg_override.py | sudo python3 # enables GFX MGCG