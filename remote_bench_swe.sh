# source .venv/bin/activate

python src/minisweagent/run/extra/swebench_remote.py \
    --server-url http://localhost:9527 \
    --output remote-swebench-o \
    --subset verified \
    --split test \
    --workers 4