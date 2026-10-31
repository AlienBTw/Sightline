pkill -INT -f "vllm serve" 2>/dev/null
for i in $(seq 1 40); do pgrep -f "vllm serve" >/dev/null || break; sleep 2; done
pgrep -af "vllm|VLLM" | grep -v pgrep || echo "vllm stopped"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader