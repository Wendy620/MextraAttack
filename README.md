Memory Privacy Attack on EHRAgent (基于 Llama-2 / Vicuna + A-mem / memos)

1. 创建工作区与项目结构

可以在任意路径下创建项目目录（如 /workspace 或 /root/EHRAgent-Attack）：

mkdir -p ~/EHR-Attack
cd ~/EHR-Attack

# 创建虚拟环境（推荐）
python3 -m venv .venv
source .venv/bin/activate

2. 安装依赖环境（必须执行）
✅ 2.1 升级 pip 并安装基础依赖
pip install --upgrade pip setuptools wheel

pip install git+https://github.com/huggingface/transformers@main   # 可选，确保 transformers 新
pip install torch tqdm
pip install hf_transfer tiktoken
pip install protobuf==3.20.*

✅ 2.2 安装本项目必需依赖
pip install sentence-transformers scikit-learn requests
pip install bitsandbytes accelerate

✅ 2.3 确保 transformers 版本 ≥ 4.44，tokenizer 等完整
python -m pip install -U \
  sentencepiece \
  "protobuf==3.20.*" \
  "transformers>=4.44.0" \
  "tokenizers>=0.19.1" \
  safetensors


  🧠 3. 安装 Memory Backend（A-mem / memos）
3.1 如果需要使用 A-mem（Agentic Memory）
git clone https://github.com/agiresearch/A-mem.git
cd A-mem
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install .


安装完可 pip list | grep amem 检查是否安装成功。

3.2 如果使用 memos 后端

无需安装服务器，只需要我们项目中的 memos_backend.py 和：

pip install sentence-transformers scikit-learn requests


可选：将记忆数据同步写入你的 memos 服务器：

export MEMOS_BASE_URL="https://<your-memos-host>"
export MEMOS_API_KEY="Bearer <your-api-token>"


4. 准备数据文件

确认你目录下包含以下两个关键文件（来自 EHRAgent 论文的数据）：

data/
 ├── general_50.json       # 50 条攻击 query
 ├── 500_solution.json     # 500 条记忆内容（Question + Knowledge + Code）


若要跑 30 条攻击，换为 general_30.json。


🚀 5. 运行攻击

export OPENAI_BASE_URL=https://api.openai-proxy.org
export OPENAI_API_KEY="your key"

export HF_TOKEN="your TOKEN"
huggingface-cli login --token "$HF_TOKEN"

bash run_all.sh

6. 完整目录结构建议
EHR-Attack/
├── data/
│   ├── general_50.json
│   ├── general_30.json （可选）
│   └── 500_solution.json
├── memory_adapters/
│   ├── memos_backend.py
│   ├── amem_backend.py      （如使用 A-mem）
│   └── factory.py
├── run_attack.py
├── evaluation.py
├── outputs/
└── README.md   ✅（即本文件）
