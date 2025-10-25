# Memory Privacy Attack on EHRAgent 
基于 Llama-2 / Vicuna + A-mem / memos

## 1. 创建工作区与项目结构
可以在任意路径下创建项目目录（如 /workspace 或 /root/EHRAgent-Attack）：
```bash
mkdir -p ~/EHR-Attack
cd ~/EHR-Attack

#  创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate
```

## 2. 安装依赖环境（必须执行）
```python
# ✅ 2.1 升级 pip 并安装基础依赖
pip install --upgrade pip setuptools wheel

# 可选，确保 transformers 较新（如果需要）

pip install git+https://github.com/huggingface/transformers@main
pip install torch tqdm
pip install hf_transfer tiktoken
pip install protobuf==3.20.*

# ✅ 2.2 安装本项目必需依赖
pip install sentence-transformers scikit-learn requests
pip install bitsandbytes accelerate

# ✅ 2.3 确保 transformers 版本 ≥ 4.44，tokenizer 等完整
python -m pip install -U \
  sentencepiece \
  "protobuf==3.20.*" \
  "transformers>=4.44.0" \
  "tokenizers>=0.19.1" \
  safetensors
```

## 3. 安装 Memory Backend（A-mem / memos）
```bash
# 3.1 如果需要使用 A-mem（Agentic Memory）
git clone https://github.com/agiresearch/A-mem.git
cd A-mem
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install .

# 3.2 如果使用 memos/amem 后端
pip install sentence-transformers scikit-learn requests
```

## 4. 准备数据文件
确认你目录下包含以下两个关键文件（来自 EHRAgent 论文的数据）：
```bash
data/
 ├── general_50.json       # 50 条攻击 query
 ├── 500_solution.json     # 500 条记忆内容（Question + Knowledge + Code）
若要跑 30 条攻击，换为 general_30.json。
```

## 5. 运行攻击 🚀
```bash
# 在运行前请设置好 OpenAI / HF 的环境变量并登录 Hugging Face：
export OPENAI_BASE_URL=https://api.openai-proxy.org
export OPENAI_API_KEY="your key"

export HF_TOKEN="your TOKEN"
huggingface-cli login --token "$HF_TOKEN"

# 然后运行：
bash run_all.sh
```

## 6. 完整目录结构建议
```bash
EHR-Attack/
├── data/
│   ├── general_50.json
│   ├── general_30.json   （可选）
│   └── 500_solution.json
├── memory_adapters/
│   ├── memos_backend.py
│   ├── amem_backend.py      （如使用 A-mem）
│   └── factory.py
├── run_attack.py
├── evaluation.py
├── outputs/
└── README.md   ✅（即本文件）
```

## 附：常见命令汇总（便于复制粘贴）
```bash
# 创建工程与虚拟环境
mkdir -p ~/EHR-Attack
cd ~/EHR-Attack
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖（示例）
pip install --upgrade pip setuptools wheel
pip install torch tqdm hf_transfer tiktoken protobuf==3.20.*
pip install sentence-transformers scikit-learn requests bitsandbytes accelerate

# transformers / tokenizers / safetensors 等
python -m pip install -U sentencepiece "protobuf==3.20.*" "transformers>=4.44.0" "tokenizers>=0.19.1" safetensors

# 如果使用 A-mem
git clone https://github.com/agiresearch/A-mem.git
cd A-mem
python -m venv .venv
source .venv/bin/activate
pip install .

# 设置环境变量并运行
export OPENAI_BASE_URL=https://api.openai-proxy.org
export OPENAI_API_KEY="your key"
export HF_TOKEN="your TOKEN"
huggingface-cli login --token "$HF_TOKEN"
bash run_all.sh
```
