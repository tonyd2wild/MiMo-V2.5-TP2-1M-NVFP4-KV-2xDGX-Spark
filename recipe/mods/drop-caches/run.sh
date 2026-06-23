#!/bin/bash
set -euo pipefail
LOG=/tmp/drop_caches.log
if pgrep -f "codex_drop_caches_loop" >/dev/null 2>&1; then
  echo "[drop-caches] loop already running"
  exit 0
fi
cat >/tmp/codex_drop_caches_loop.sh <<'EOF'
#!/bin/bash
while true; do
  sync
  echo 1 >/proc/sys/vm/drop_caches 2>/dev/null || true
  sleep 60
done
EOF
chmod +x /tmp/codex_drop_caches_loop.sh
nohup /tmp/codex_drop_caches_loop.sh >>"$LOG" 2>&1 &
echo "[drop-caches] started loop pid=$! log=$LOG"
