# -*- coding: utf-8 -*-
import os
import shutil
import zipfile

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from ..JSON2YOLO.general_json2yolo import convert_coco_json

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data...')

    temp_data_path = "../../data/interim"
    processed_data_path = "../../data/processed"

    for root, dirs, files in os.walk(temp_data_path):
        for f in files:
            os.remove(os.path.join(root, f))

        for d in dirs:
            shutil.rmtree(os.path.join(temp_data_path, d))

    for root, dirs, files in os.walk(input_filepath):
        for f in files:
            if f.endswith('.zip'):
                with zipfile.ZipFile(os.path.join(root, f)) as zip_ref:
                    zip_ref.extractall(temp_data_path)

    logger.info("Converting COCO json labels into YOLO format...")
    convert_coco_json(temp_data_path)

    full_data_path: str = os.path.join(temp_data_path, "__full_data")
    extension_allowed: str = '.jpg'
    split_percentage: int = 80

    logger.info("Searching for images...")
    # Move images from obj directory to full_data directory
    for r, d, f in os.walk(temp_data_path):
        for file in f:
            if file.endswith(extension_allowed):
                shutil.move(os.path.join(r, file), full_data_path)
    
    images_path = os.path.join(processed_data_path, "images")
    if os.path.exists(images_path):
        shutil.rmtree(images_path)
    os.mkdir(images_path)

    labels_path = os.path.join(processed_data_path, "images")
    if os.path.exists(labels_path):
        shutil.rmtree(labels_path)
    os.mkdir(labels_path)

    training_images_path = os.path.join(images_path, 'training')
    validation_images_path = os.path.join(images_path, 'validation')
    training_labels_path = os.path.join(labels_path, 'training')
    validation_labels_path = os.path.join(labels_path, 'validation')

    os.mkdir(training_images_path)
    os.mkdir(validation_images_path)
    os.mkdir(training_labels_path)
    os.mkdir(validation_labels_path)

    files = []

    ext_len = len(extension_allowed)

    for r, d, f in os.walk(full_data_path):
        for file in f:
            if file.endswith(extension_allowed):
                strip = file[0:len(file) - ext_len]
                files.append(strip)

    random.shuffle(files)
    size = len(files)
    split = int(split_percentage * size / 100)

    logger.info("Copying training data...")
    for i in range(split):
        strip = files[i]

        image_file = strip + extension_allowed
        shutil.copy(os.path.join(full_data_path, image_file), training_images_path)

        annotation_file = strip + '.txt'
        shutil.copy(os.path.join(full_data_path, annotation_file), training_labels_path)

    logger.info("Copying validation data...")
    for i in range(split, size):
        strip = files[i]

        image_file = strip + extension_allowed
        shutil.copy(os.path.join(full_data_path, image_file), validation_images_path)

        annotation_file = strip + '.txt'
        shutil.copy(os.path.join(full_data_path, annotation_file), validation_labels_path)
    logger.info("Finished!")

    # find number of classes

    project_dir = Path(__file__).resolve().parents[2]
    with open(f'{project_dir}/dataset.yaml', 'w') as f:
        f.write(f'train: {training_images_path}\n')
        f.write(f'val: {validation_images_path}\n')
        f.write('nc: 4\n')
        f.write("names: ['pet', 'ps', 'pp', 'pe']")
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
