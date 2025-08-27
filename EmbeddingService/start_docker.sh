docker run -itd \
    --gpus 'device=0' \
    --restart always \
    -p 7100:8000 \
    -v /disk2/Models/jina-embeddings-v4-vllm-text-matching:/models \
    --name jina-embeddings-v4-vllm \
    vllm/vllm-openai:v0.9.2 \
    --model /models \
    --served-model-name jina-embeddings-v4 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --max-model-len 32768 \
    --task embed
