#!/bin/bash

WORK_PATH=`pwd`

# sceneflow/flyingthings3d
DATA_SUBSET_PATH="${WORK_PATH}/datasets/SceneFlow/FlyingThings3D"
mkdir -p ${DATA_SUBSET_PATH} && cd ${DATA_SUBSET_PATH}
# flyingthings3d__frames_cleanpass_webp.tar
wget -t 0 https://academictorrents.com/download/d20b0f88033b652b84ef4fa49ebcaa7f692df1a5.torrent
transmission-cli d20b0f88033b652b84ef4fa49ebcaa7f692df1a5.torrent -w ${DATA_SUBSET_PATH}
tar -xvf flyingthings3d__frames_cleanpass_webp.tar
rm *.torrent
rm *.tar
