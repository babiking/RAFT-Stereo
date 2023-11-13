#!/bin/bash

WORK_PATH=`pwd`
DATA_PATH="/mnt/data/workspace/datasets"

echo "[START] SceneFlow/FlyingThings3D dataset downloading..."
# rm ~/.config/transmission/resume/torrentyouwanttorestart.resume
mkdir -p ${DATA_PATH}/SceneFlow/FlyingThings3D
cd ${DATA_PATH}/SceneFlow/FlyingThings3D
wget -t 0 https://academictorrents.com/download/d20b0f88033b652b84ef4fa49ebcaa7f692df1a5.torrent
transmission-cli d20b0f88033b652b84ef4fa49ebcaa7f692df1a5.torrent -w ${DATA_PATH}/SceneFlow/FlyingThings3D

wget -t 0 https://academictorrents.com/download/3221ff49a08f5e6749f24958c1f76248fe9cb884.torrent
transmission-cli 3221ff49a08f5e6749f24958c1f76248fe9cb884.torrent -w ${DATA_PATH}/SceneFlow/FlyingThings3D
echo "[FINISH] SceneFlow/FlyingThings3D dataset downloaded."
cd ${WORK_PATH}



echo "[START] SceneFlow/Driving dataset downloading..."
mkdir -p ${DATA_PATH}/SceneFlow/Driving
cd ${DATA_PATH}/SceneFlow/Driving
wget -t 0 https://academictorrents.com/download/a52319069b8e8d4d8b4ada6251f031ed8cf7cecc.torrent
transmission-cli a52319069b8e8d4d8b4ada6251f031ed8cf7cecc.torrent -w ${DATA_PATH}/SceneFlow/Driving

wget -t 0 https://academictorrents.com/download/1d642a371312d193ae4523e089bf127917294175.torrent
transmission-cli 1d642a371312d193ae4523e089bf127917294175.torrent -w ${DATA_PATH}/SceneFlow/Driving
echo "[FINISH] SceneFlow/Driving dataset downloaded."
cd ${WORK_PATH}
