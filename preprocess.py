import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageEnhance
from skimage.io import imsave, imread
from sklearn.model_selection import train_test_split

from helpers import (
    find_ct_path,
    get_patient_table,
    get_nodule_mask,
    get_series_uid,
    get_lung_mask,
    normalize,
    resize
)


def extract(sourcepath='data/raw/', destpath='data/extracted', k=4):
    """
    Given path to raw data and an output path, extracts and saves the
    2D slices and labels from the raw LIDC-IDRI data

    :param sourcepath: path to raw data
    :param destpath: path to directory to save extracted data
    :param k: the max number of non-tumor and tumor slices to take from
              each patient (2k total slices from each patient)
    :return: None
    """

    # construct directory for extracted files
    os.mkdir(f"{destpath}")
    os.mkdir(f"{destpath}/image")
    os.mkdir(f"{destpath}/mask")
    os.mkdir(f"{destpath}/label")

    # tracks which images belong to which patient (for later test/train split)
    ImgLookup = {}

    start = 1
    end = len(os.listdir(f"{sourcepath}/LIDC-IDRI/"))

    id_nums = [
        '0'*(4-len(n))+n for n in [str(i) for i in range(start, end+1)]
    ]
    ids = [f"LIDC-IDRI/LIDC-IDRI-{id_num}" for id_num in id_nums]

    for i, patient_id in enumerate(ids):
        sys.stdout.write(f"\rExtracting...{i+1}/{end+1-start}")
        sys.stdout.flush()
        # check if patient in LUMA
        uids = pickle.load(open("uids.pkl", "rb"))
        if not os.path.exists(sourcepath + patient_id):
            continue
        if get_series_uid(find_ct_path(sourcepath, patient_id)) not in uids:
            continue
        ImgLookup[patient_id] = []
        # get image and contours for patient images
        slices_table = get_patient_table(sourcepath, patient_id, k=k)
        images = []
        masks = []
        labels = []
        for row in slices_table.iterrows():
            path, rois = row[1].path, row[1].ROIs
            try:
                if pd.isna(rois):
                    labels.append(False)
            except ValueError:
                labels.append(True)
            im = pydicom.dcmread(path).pixel_array
            masks.append(get_nodule_mask(im, rois))
            images.append(im)

        # save prepared image and mask in properly constructed directory
        for im, mask, label in zip(images, masks, labels):
            idx = len(os.listdir(f"{destpath}/image/"))
            imsave(f"{destpath}/image/{idx}.tif", im)
            imsave(f"{destpath}/mask/{idx}.tif", mask)
            pickle.dump(label, open(f"{destpath}/label/{idx}.pkl", 'wb'))
            ImgLookup[patient_id].append(idx)
    pickle.dump(ImgLookup, open(f"data/ImageLookup.pkl", 'wb'))


def preprocess(sourcepath='data/extracted', destpath='data/processed'):
    os.mkdir(destpath)
    os.mkdir(f"{destpath}/image")
    os.mkdir(f"{destpath}/mask")
    os.mkdir(f"{destpath}/label")

    idxs = range(len(os.listdir(f"{sourcepath}/image/")))
    n = len(idxs)
    for i, idx in enumerate(idxs):
        sys.stdout.write(f"\rProcessing...{i+1}/{n}")
        sys.stdout.flush()
        img = imread(f"{sourcepath}/image/{idx}.tif")
        lung_mask = resize(
            get_lung_mask(img).astype('float')
        )
        img = normalize(img)
        img = resize(img)
        img = img*lung_mask
        img = img*lung_mask
        pil_im = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(pil_im)
        enhanced_im = enhancer.enhance(2.0)
        np_im = np.array(enhanced_im)
        imsave(f"{destpath}/image/{idx}.tif", np_im)

        mask = imread(f"{sourcepath}/mask/{idx}.tif")
        mask = resize(mask)
        imsave(f"{destpath}/mask/{idx}.tif", mask)

        lbl_source = f"{sourcepath}/label/{idx}.pkl"
        lbl_dest = f"{destpath}/label/{i}.pkl"
        shutil.copyfile(lbl_source, lbl_dest)
    print(f"\nComplete.")


def test_train_split(
    sourcepath='data/processed',
    trainpath='data/train',
    testpath='data/test',
    valpath='data/val'
):
    """
    Called from download_transform_split. Creates training and test sets from
    prepared images

    :param datapath: the directory containing prepared, train, and test folders
    :return: None
    """
    for d in [trainpath, testpath, valpath]:
        os.mkdir(f"{d}")
        os.mkdir(f"{d}/image")
        os.mkdir(f"{d}/label")
        os.mkdir(f"{d}/mask")

    ImageLookup = pickle.load(open("data/ImageLookup.pkl", "rb"))

    ids = list(ImageLookup.keys())

    train_ids, test_ids = train_test_split(ids, test_size=.2)
    train_ids, val_ids = train_test_split(train_ids, test_size=.1)

    count = 0
    for id_num in train_ids:
        idxs = [
            i for i in ImageLookup[id_num]
            if os.path.exists(f"{sourcepath}/image/{i}.tif")
        ]
        for idx in idxs:
            im_source = f"{sourcepath}/image/{idx}.tif"
            im_dest = f"{trainpath}/image/{count}.tif"
            shutil.copyfile(im_source, im_dest)

            lbl_source = f"{sourcepath}/label/{idx}.pkl"
            lbl_dest = f"{trainpath}/label/{count}.pkl"
            shutil.copyfile(lbl_source, lbl_dest)

            msk_source = f"{sourcepath}/mask/{idx}.tif"
            msk_dest = f"{trainpath}/mask/{count}.tif"
            shutil.copy(msk_source, msk_dest)
            count += 1

    count = 0
    for id_num in val_ids:
        idxs = [
            i for i in ImageLookup[id_num]
            if os.path.exists(f"{sourcepath}/image/{i}.tif")
        ]
        for idx in idxs:
            im_source = f"{sourcepath}/image/{idx}.tif"
            im_dest = f"{valpath}/image/{count}.tif"
            shutil.copyfile(im_source, im_dest)

            lbl_source = f"{sourcepath}/label/{idx}.pkl"
            lbl_dest = f"{valpath}/label/{count}.pkl"
            shutil.copyfile(lbl_source, lbl_dest)

            msk_source = f"{sourcepath}/mask/{idx}.tif"
            msk_dest = f"{valpath}/mask/{count}.tif"
            shutil.copy(msk_source, msk_dest)
            count += 1

    count = 0
    for id_num in test_ids:
        idxs = [
            i for i in ImageLookup[id_num]
            if os.path.exists(f"{sourcepath}/image/{i}.tif")
        ]
        for idx in idxs:
            im_source = f"{sourcepath}/image/{idx}.tif"
            im_dest = f"{testpath}/image/{count}.tif"
            shutil.copyfile(im_source, im_dest)

            lbl_source = f"{sourcepath}/label/{idx}.pkl"
            lbl_dest = f"{testpath}/label/{count}.pkl"
            shutil.copyfile(lbl_source, lbl_dest)

            msk_source = f"{sourcepath}/mask/{idx}.tif"
            msk_dest = f"{testpath}/mask/{count}.tif"
            shutil.copy(msk_source, msk_dest)
            count += 1
