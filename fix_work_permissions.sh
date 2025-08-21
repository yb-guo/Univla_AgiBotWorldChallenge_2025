#!/bin/bash
set -e

echo "🔧 修复 /work 下的目录权限（自动跳过挂载目录）..."

# 显式跳过的目录名（按需增减）
SKIP_NAMES=("data" "shared_outputs" "outputs" "weights")

is_in_skip_list() {
  local name="$1"
  for s in "${SKIP_NAMES[@]}"; do
    if [[ "$name" == "$s" ]]; then
      return 0
    fi
  done
  return 1
}

is_fuse_mount() {
  local path="$1"
  # 方式1：看文件系统类型
  local fstype
  fstype=$(stat -f -c %T "$path" 2>/dev/null || echo "unknown")
  case "$fstype" in
    fuse|fuseblk|fuse.blobfuse2|fuse.azureblob|blobfuse|blobfuse2)
      return 0;;
  esac

  # 方式2：从 /proc/mounts 判断
  if grep -qsE " $path | $(readlink -f "$path") " /proc/mounts; then
    if grep -qsE "(fuse|blobfuse)" /proc/mounts; then
      # 粗略：任何 fuse/blobfuse 命中都跳过
      return 0
    fi
  fi
  return 1
}

for d in /work/*; do
  base=$(basename "$d")
  if is_in_skip_list "$base"; then
    echo "⏩ 按名称跳过：$d"
    continue
  fi
  if is_fuse_mount "$d"; then
    echo "⏩ 按挂载类型跳过（fuse/blobfuse）：$d"
    continue
  fi

  echo "👉 正在修改 $d ..."
  sudo chown -R aiscuser "$d"
done

echo "✅ 权限修复完成"
