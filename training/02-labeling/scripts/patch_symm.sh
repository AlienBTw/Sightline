set -e
F=$HOME/sglang-venv/lib/python3.12/site-packages/sglang/srt/distributed/device_communicators/triton_symm_mem_ag.py
[ -e $F.orig ] || cp $F $F.orig
grep -q SIGHTLINE_PATCH $F || python3 - "$F" <<'EOF'
import sys
p=sys.argv[1]; s=open(p).read()
old="        self._state = self._UNINIT if enabled else None\n"
new=old+"        # SIGHTLINE_PATCH: symmetric-memory rendezvous dies with SIGFPE on 2x3090 under WSL (no P2P); force NCCL.\n        import os as _os\n        if _os.environ.get(\"SGLANG_DISABLE_MULTIMEM_AG\") == \"1\":\n            self._state = None\n"
assert s.count(old)==1; open(p,"w").write(s.replace(old,new))
EOF
grep -n -A3 SIGHTLINE_PATCH $F