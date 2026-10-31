set -e
D=$HOME/sglang-venv/lib/python3.12/site-packages/nvidia/cu13
[ -e $D/lib64 ] || ln -s lib $D/lib64
[ -e $D/lib/libcudart.so ] || ln -s libcudart.so.13 $D/lib/libcudart.so
mkdir -p $D/lib/stubs
[ -e $D/lib/stubs/libcuda.so ] || ln -s /usr/lib/wsl/lib/libcuda.so $D/lib/stubs/libcuda.so
ls -l $D/lib64 $D/lib/libcudart.so $D/lib/stubs/; ls -l /usr/lib/wsl/lib/libcuda.so*