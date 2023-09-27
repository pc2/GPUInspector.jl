_device2string(dev::HIPDevice) = "GPU $(_gpuid(dev)): $(_name(dev))"

_gpuid(dev::HIPDevice) = AMDGPU.HIP.device_id(dev) + 1

_name(dev::HIPDevice) = AMDGPU.HIP.name(dev)
