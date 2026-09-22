#!/usr/bin/env python3
"""无需安装 PyTorch，将 bin checkpoint 转为 safetensors。"""
import os
import sys
import zipfile
import pickle
import numpy as np
import safetensors.numpy as snp

class TorchUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return lambda storage, storage_offset, size, stride, requires_grad, backward_hooks: (storage, storage_offset, size, stride)
        if module == 'torch._tensor' and name == '_rebuild_from_type_v2':
            return lambda func, new_type, args, state: func(*args)
        return type(name, (), {'__module__': module})

    def persistent_load(self, pid):
        return ('pid', pid)

def convert(bin_path, out_path):
    print(f"Reading {bin_path}...")
    with zipfile.ZipFile(bin_path) as z:
        pkl_name = [n for n in z.namelist() if n.endswith('data.pkl')][0]
        prefix = os.path.dirname(pkl_name)
        with z.open(pkl_name) as f:
            unpickler = TorchUnpickler(f)
            state_dict = unpickler.load()

        tensors = {}
        for name, v in state_dict.items():
            storage, offset, size, stride = v
            storage_id = storage[1][2]
            raw_data_name = f"{prefix}/data/{storage_id}"
            buf = z.read(raw_data_name)
            arr = np.frombuffer(buf, dtype=np.float32).reshape(size)
            tensors[name] = arr

        print(f"Saving {len(tensors)} tensors to {out_path}...")
        snp.save_file(tensors, out_path)
        print(f"Successfully converted {bin_path} -> {out_path} ({os.path.getsize(out_path)/(1024*1024):.1f} MB)")

if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else '/home/aiza/workspace/assets/ai-models/TinyLlama_v1.1/pytorch_model.bin'
    dst = sys.argv[2] if len(sys.argv) > 2 else '/home/aiza/workspace/assets/ai-models/TinyLlama_v1.1/model.safetensors'
    convert(src, dst)
