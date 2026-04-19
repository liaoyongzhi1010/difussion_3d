#!/usr/bin/env bash
# Unpack the EULA-downloaded 3D-FRONT archives into the canonical layout.
# Alibaba does not publish a programmatic download API; the user must fetch
# the three zips manually (see datasets/README.md section 1).
set -euo pipefail

TARGET="${1:-data/external/3d_front}"

if [[ ! -d "${TARGET}" ]]; then
    echo "creating ${TARGET}"
    mkdir -p "${TARGET}"
fi

for archive in 3D-FRONT.zip 3D-FRONT-texture.zip 3D-FUTURE-model.zip; do
    if [[ ! -f "${TARGET}/${archive}" ]]; then
        echo "missing ${TARGET}/${archive}"
        echo "download from https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future after signing EULA"
        exit 1
    fi
done

cd "${TARGET}"
for archive in 3D-FRONT.zip 3D-FRONT-texture.zip 3D-FUTURE-model.zip; do
    name="${archive%.zip}"
    if [[ -d "${name}" ]]; then
        echo "${name}/ already extracted, skipping"
        continue
    fi
    echo "extracting ${archive} ..."
    unzip -q "${archive}"
done
echo "done. expected subdirs: 3D-FRONT/ 3D-FRONT-texture/ 3D-FUTURE-model/"
