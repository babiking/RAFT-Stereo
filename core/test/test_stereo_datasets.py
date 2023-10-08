import pytest

import os
import time
import cv2 as cv
from tqdm import tqdm
from core.stereo_datasets import ETH3D, KITTI, Middlebury


def get_work_path():
    work_path = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "../.."))
    return work_path


def get_augment_parameters():
    aug_params = {
        "crop_size": [240, 320],
        "min_scale": -0.2,
        "max_scale": 0.4,
        "saturation_range": [0.0, 1.4],
        "do_flip": False,
        "yjitter": True,
    }
    return aug_params


def test_middlebury_Q_dataset():
    work_path = get_work_path()

    aug_params = get_augment_parameters()

    dataset = Middlebury(aug_params=aug_params,
                         root="/home/ec2-user/datasets/Middlebury",
                         split="Q")

    assert len(dataset) == 15, \
        f"wrong # of items Middlebury_Q/train: 15 VS ({len(dataset)})!"

    pbar = tqdm(total=len(dataset))
    for i in range(len(dataset)):
        start = time.time()
        img_file_list, l_img, r_img, l_disp, _ = dataset[i]
        end = time.time()

        elapsed = float(end - start)
        assert elapsed < 0.8, \
            f"timeout ({elapsed:.4f} sec) for Middlebury_Q data item iteration!"

        assert l_img.cpu().detach().numpy().shape == (3, 240, 320)
        assert r_img.cpu().detach().numpy().shape == (3, 240, 320)
        assert l_disp.cpu().detach().numpy().shape == (1, 240, 320)

        pbar.set_description(
            f"{os.path.relpath(img_file_list[0], work_path)}.")
        pbar.update(1)


def test_eth3d_dataset():
    work_path = get_work_path()

    aug_params = get_augment_parameters()

    dataset = ETH3D(aug_params=aug_params,
                    root="/home/ec2-user/datasets/ETH3D",
                    split="training")

    assert len(dataset) == 27, \
        f"wrong # of items ETH3D/train: 27 VS ({len(dataset)})!"

    pbar = tqdm(total=len(dataset))
    for i in range(len(dataset)):
        start = time.time()
        img_file_list, l_img, r_img, l_disp, _ = dataset[i]
        end = time.time()

        elapsed = float(end - start)
        assert elapsed < 0.8, \
            f"timeout ({elapsed:.4f} sec) for ETH3D data item iteration!"

        assert l_img.cpu().detach().numpy().shape == (3, 240, 320)
        assert r_img.cpu().detach().numpy().shape == (3, 240, 320)
        assert l_disp.cpu().detach().numpy().shape == (1, 240, 320)

        pbar.set_description(
            f"{os.path.relpath(img_file_list[0], work_path)}.")
        pbar.update(1)


def test_kitti_dataset():
    work_path = get_work_path()

    aug_params = get_augment_parameters()

    dataset = KITTI(aug_params=aug_params,
                    root="/home/ec2-user/datasets/KITTI",
                    image_set="training")

    assert len(dataset) == 200, \
        f"wrong # of items KITTI/train: 200 VS ({len(dataset)})!"

    pbar = tqdm(total=len(dataset))
    for i in range(len(dataset)):
        start = time.time()
        img_file_list, l_img, r_img, l_disp, _ = dataset[i]
        end = time.time()

        elapsed = float(end - start)
        assert elapsed < 0.8, \
            f"timeout ({elapsed:.4f} sec) for KITTI data item iteration!"

        assert l_img.cpu().detach().numpy().shape == (3, 240, 320)
        assert r_img.cpu().detach().numpy().shape == (3, 240, 320)
        assert l_disp.cpu().detach().numpy().shape == (1, 240, 320)

        pbar.set_description(
            f"{os.path.relpath(img_file_list[0], work_path)}.")
        pbar.update(1)


def main():
    test_middlebury_Q_dataset()
    test_eth3d_dataset()
    test_kitti_dataset()


if __name__ == "__main__":
    main()