# Sightline: stop the SGLang server in WSL (SIGINT, then wait up to 80 s).
# The original used a hard-coded PID (`kill -INT 367`); this copy matches the process by name instead.
pkill -INT -f sglang.launch_server 2>/dev/null
for i in $(seq 1 40); do pgrep -f sglang.launch_server >/dev/null || break; sleep 2; done
pgrep -af "sglang" || echo "sglang stopped"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader