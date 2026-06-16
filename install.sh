#!/bin/bash
set -e

PYTHON="${PYTHON:-python3}"

# 检查 Python 版本
PY_VERSION=$($PYTHON --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [ "$PY_VERSION" != "3.12" ]; then
    echo "错误: 需要 Python 3.12，当前为 $PY_VERSION"
    echo "请设置环境变量 PYTHON 指向 Python 3.12: PYTHON=python3.12 bash install.sh"
    exit 1
fi

# 虚拟环境目录配置
VENV_DIR="${VENV_DIR:-$HOME/.local/share/xiaoming/venv}"
BIN_DIR="$HOME/.local/bin"

echo "=== 安装 Xiaoming ==="
echo "Python: $($PYTHON --version)"
echo "虚拟环境: $VENV_DIR"

# 创建虚拟环境
if [ ! -d "$VENV_DIR" ]; then
    echo "创建虚拟环境..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# 在虚拟环境中安装
echo "安装依赖..."
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -e "$(dirname "$0")"

# 创建可执行脚本到 ~/.local/bin
mkdir -p "$BIN_DIR"

for cmd in xiaoming xiaoming-cli xiaoming-eval; do
    cat > "$BIN_DIR/$cmd" << SCRIPT
#!/bin/bash
exec "$VENV_DIR/bin/$cmd" "\$@"
SCRIPT
    chmod +x "$BIN_DIR/$cmd"
done

echo ""
echo "安装完成!"
echo ""
echo "请确保 ~/.local/bin 在你的 PATH 中:"
echo '  export PATH="$HOME/.local/bin:$PATH"'
echo ""
echo "然后可以直接使用:"
echo "  xiaoming --help"
echo "  xiaoming-cli --help"
echo "  xiaoming-eval --help"
