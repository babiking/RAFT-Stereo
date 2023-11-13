#!/bin/bash

WORK_PATH=`pwd`
DATA_PATH="/home/babiking/datasets"

echo "[START] SceneFlow/FlyingThings3D dataset downloading..."
# rm ~/.config/transmission/resume/torrentyouwanttorestart.resume
mkdir -p ${DATA_PATH}/SceneFlow/FlyingThings3D
cd ${DATA_PATH}/SceneFlow/FlyingThings3D
wget -t 0 https://academictorrents.com/download/20afbe18b5d1754b75deeefe4c2c74b17c9ea792.torrent
transmission-cli 20afbe18b5d1754b75deeefe4c2c74b17c9ea792.torrent -w ${DATA_PATH}/SceneFlow/FlyingThings3D

wget -t 0 https://academictorrents.com/download/3221ff49a08f5e6749f24958c1f76248fe9cb884.torrent
transmission-cli 3221ff49a08f5e6749f24958c1f76248fe9cb884.torrent -w ${DATA_PATH}/SceneFlow/FlyingThings3D
echo "[FINISH] SceneFlow/FlyingThings3D dataset downloaded."
cd ${WORK_PATH}



echo "[START] SceneFlow/Driving dataset downloading..."
mkdir -p ${DATA_PATH}/SceneFlow/Driving
cd ${DATA_PATH}/SceneFlow/Driving
wget -t 0 https://academictorrents.com/download/ea392433e3dfcb4b83dcd3300dfa9b9919ef8e1f.torrent
transmission-cli ea392433e3dfcb4b83dcd3300dfa9b9919ef8e1f.torrent -w ${DATA_PATH}/SceneFlow/Driving

wget -t 0 https://academictorrents.com/download/1d642a371312d193ae4523e089bf127917294175.torrent
transmission-cli 1d642a371312d193ae4523e089bf127917294175.torrent -w ${DATA_PATH}/SceneFlow/Driving
echo "[FINISH] SceneFlow/Driving dataset downloaded."
cd ${WORK_PATH}
