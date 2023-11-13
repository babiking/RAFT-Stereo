#!/bin/bash

WORK_PATH=`pwd`
DATA_PATH="/mnt/data/workspace/datasets"

echo "[START] Middlebury dataset downloading..."
mkdir -p ${DATA_PATH}/Middlebury
cd ${DATA_PATH}/Middlebury
wget https://www.dropbox.com/s/fn8siy5muak3of3/official_train.txt -P MiddEval3/
wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip
unzip MiddEval3-data-Q.zip
wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip
unzip MiddEval3-GT0-Q.zip
wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-H.zip
unzip MiddEval3-data-H.zip
wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-H.zip
unzip MiddEval3-GT0-H.zip
wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-F.zip
unzip MiddEval3-data-F.zip
wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-F.zip
unzip MiddEval3-GT0-F.zip
rm *.zip
echo "[FINISH] Middlebury dataset downloaded."
cd ${WORK_PATH}



echo "[START] Middlebury-2014 dataset downloading..."
mkdir -p ${DATA_PATH}/Middlebury/2014
cd ${DATA_PATH}/Middlebury/2014
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Adirondack-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Adirondack-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Backpack-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Backpack-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Bicycle1-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Bicycle1-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Cable-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Cable-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Classroom1-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Classroom1-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Couch-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Couch-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Flowers-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Flowers-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Jadeplant-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Jadeplant-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Mask-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Mask-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Motorcycle-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Motorcycle-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Piano-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Piano-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Pipes-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Pipes-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Playroom-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Playroom-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Playtable-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Playtable-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Recycle-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Recycle-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Shelves-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Shelves-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Shopvac-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Shopvac-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Sticks-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Sticks-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Storage-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Storage-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Sword1-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Sword1-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Sword2-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Sword2-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Umbrella-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Umbrella-perfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Vintage-imperfect.zip
wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Vintage-perfect.zip
unzip \*.zip
rm -vf *.zip
echo "[FINISH] Middlebury-2014 dataset downloaded."
cd ${WORK_PATH}



echo "[START] ETH3D dataset downloading..."
mkdir -p ${DATA_PATH}/ETH3D/two_view_training
cd ${DATA_PATH}/ETH3D/two_view_training
wget https://www.eth3d.net/data/two_view_training.7z
echo "Unzipping two_view_training.7z using p7zip (installed from environment.yaml)"
7za x two_view_training.7z
rm two_view_training.7z

mkdir -p ${DATA_PATH}/ETH3D/two_view_training_gt
cd ${DATA_PATH}/ETH3D/two_view_training_gt
wget https://www.eth3d.net/data/two_view_training_gt.7z
echo "Unzipping two_view_training_gt.7z using p7zip (installed from environment.yaml)"
7za x two_view_training_gt.7z
rm two_view_training_gt.7z

mkdir -p ${DATA_PATH}/ETH3D/two_view_testing
cd ${DATA_PATH}/ETH3D/two_view_testing
wget https://www.eth3d.net/data/two_view_test.7z
echo "Unzipping two_view_test.7z using p7zip (installed from environment.yaml)"
7za x two_view_test.7z
rm two_view_test.7z
cd ${WORK_PATH}
echo "[FINISH] ETH3D dataset downloaded."