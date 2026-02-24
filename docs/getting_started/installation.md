### Environment Setup

**Clone the repository**
```bash
# 官方仓库
git clone https://github.com/0russwest0/Agent-R1.git
# 或使用你自己的 fork/仓库
# git clone https://github.com/YOUR_USERNAME/Agent-R1.git
cd Agent-R1
```

**Install `verl`**
```bash
# Create the conda environment
conda create -n verl python==3.10
conda activate verl

# install verl together with some lightweight dependencies in setup.py
git submodule update --init --recursive
cd verl
pip3 install -e .

# Install the latest stable version of vLLM
pip3 install vllm

# Install flash-attn
pip3 install flash-attn --no-build-isolation
```