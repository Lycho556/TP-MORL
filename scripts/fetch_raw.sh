#!/usr/bin/env bash
# 拉取 pSO 原始栅格（144 MB，不入 git）。数据来源: https://github.com/codeRimoe/pSO
set -e
cd "$(dirname "$0")/../data/raw"
[ -d pso ] || git clone --depth 1 https://github.com/codeRimoe/pSO.git pso
echo "raw rasters at data/raw/pso/pMOLU/GMCase/"
