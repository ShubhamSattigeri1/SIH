#!/usr/bin/env python3
"""
LandCover.AI Segmentation Model for Carbon Credits Estimation
Complete pipeline for training, inference, and carbon credit calculation
"""

import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import albumentations as A
from sklearn.model_selection import train_test_split
from skimage import morphology
from scipy import ndimage
import json
from typing import Tuple, List, Dict
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class LandCoverConfig:
    DATASET_PATH = "./landcover_ai"
    IMAGES_PATH = os.path.join(DATASET_PATH, "images")
    MASKS_PATH = os.path.join(DATASET_PATH, "masks")
    
    IMG_HEIGHT = 512
    IMG_WIDTH = 512
    NUM_CLASSES = 8
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 1e-4

    CLASS_NAMES = {
        0: 'background',
        1: 'buildings', 
        2: 'woodland',
        3: 'water',
        4: 'road',
        5: 'railway',
        6: 'vegetation',
        7: 'cars'
    }
    
    VEGETATION_CLASS = 6
    DEFAULT_CARBON_RATE = 4.0

class ImageProcessor:
    @staticmethod
    def handle_rgba_image(image: np.ndarray, background_color=(255,255,255)) -> np.ndarray:
        if image.shape[-1] != 4:
            return image[:, :, :3]
        rgb = image[:, :, :3].astype(np.float32) / 255.0
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        background = np.array(background_color).reshape(1,1,3)/255.0
        composited = alpha * rgb + (1-alpha) * background
        return (composited*255).astype(np.uint8)
    
    @staticmethod
    def preprocess_image(image_path: str, target_size: Tuple[int,int]) -> np.ndarray:
        pil_image = Image.open(image_path)
        image = np.array(pil_image)
        if len(image.shape)==2:  # grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[-1]==4:
            image = ImageProcessor.handle_rgba_image(image)
        image = cv2.resize(image, target_size)
        return image.astype(np.float32)/255.0

class DataGenerator(keras.utils.Sequence):
    def _init_(self, image_paths: List[str], mask_paths: List[str], 
                 batch_size: int, img_size: Tuple[int,int], 
                 num_classes: int, augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.augment = augment
        self.indices = np.arange(len(image_paths))
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ])
        else:
            self.transform = None
        self.on_epoch_end()
    
    def _len_(self):
        return len(self.image_paths)//self.batch_size
    
    def _getitem_(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        return self._generate_batch(indices)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
    
    def _generate_batch(self, indices):
        batch_images = np.zeros((len(indices), *self.img_size, 3), dtype=np.float32)
        batch_masks = np.zeros((len(indices), *self.img_size, self.num_classes), dtype=np.float32)
        for i, idx in enumerate(indices):
            image = ImageProcessor.preprocess_image(self.image_paths[idx], self.img_size)
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.img_size)
            if self.transform:
                augmented = self.transform(image=(image*255).astype(np.uint8), mask=mask)
                image = augmented['image'].astype(np.float32)/255.0
                mask = augmented['mask']
            batch_images[i] = image
            batch_masks[i] = keras.utils.to_categorical(mask, self.num_classes)
        return batch_images, batch_masks

class UNetModel:
    @staticmethod
    def conv_block(x, filters, kernel_size=3, padding='same'):
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x
    
    @staticmethod
    def encoder_block(x, filters):
        skip = x
        x = UNetModel.conv_block(x, filters)
        x = UNetModel.conv_block(x, filters)
        if skip.shape[-1] != filters:
            skip = layers.Conv2D(filters, 1)(skip)
        x = layers.Add()([x, skip])
        return x
    
    @staticmethod
    def decoder_block(x, skip, filters):
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([x, skip])
        x = UNetModel.conv_block(x, filters)
        x = UNetModel.conv_block(x, filters)
        return x
    
    @staticmethod
    def build_model(input_shape, num_classes):
        inputs = layers.Input(input_shape)
        e1 = UNetModel.conv_block(inputs, 64)
        e1 = UNetModel.encoder_block(e1, 64)
        p1 = layers.MaxPooling2D(2)(e1)
        e2 = UNetModel.encoder_block(p1, 128)
        p2 = layers.MaxPooling2D(2)(e2)
        e3 = UNetModel.encoder_block(p2, 256)
        p3 = layers.MaxPooling2D(2)(e3)
        e4 = UNetModel.encoder_block(p3, 512)
        p4 = layers.MaxPooling2D(2)(e4)
        b = UNetModel.encoder_block(p4, 1024)
        d4 = UNetModel.decoder_block(b, e4, 512)
        d3 = UNetModel.decoder_block(d4, e3, 256)
        d2 = UNetModel.decoder_block(d3, e2, 128)
        d1 = UNetModel.decoder_block(d2, e1, 64)
        outputs = layers.Conv2D(num_classes, 1, activation='softmax')(d1)
        return keras.Model(inputs, outputs, name='UNet_ResNet_Encoder')

class LandCoverSegmentationModel:
    def _init_(self, config=None):
        self.config = config or LandCoverConfig()
        self.model = None
    
    def prepare_dataset(self):
        exts = ('.jpg','.jpeg','.png','.tif','.tiff')
        image_files = [f for f in os.listdir(self.config.IMAGES_PATH) if f.lower().endswith(exts)]
        image_paths, mask_paths = [], []
        for f in image_files:
            img_path = os.path.join(self.config.IMAGES_PATH,f)
            mask_name = os.path.splitext(f)[0]+os.path.splitext(f)[1]
            mask_path = os.path.join(self.config.MASKS_PATH, mask_name)
            if os.path.exists(mask_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path)
        print(f"Found {len(image_paths)} image-mask pairs")
        if len(image_paths)==0:
            raise ValueError("No image-mask pairs found. Check dataset paths!")
        train_images, val_images, train_masks, val_masks = train_test_split(
            image_paths, mask_paths, test_size=0.2, random_state=42
        )
        print(f"Training samples: {len(train_images)}, Validation samples: {len(val_images)}")
        return train_images, train_masks, val_images, val_masks
    
    def build_and_compile_model(self):
        input_shape = (self.config.IMG_HEIGHT, self.config.IMG_WIDTH, 3)
        self.model = UNetModel.build_model(input_shape, self.config.NUM_CLASSES)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"Model compiled, total parameters: {self.model.count_params():,}")
        return self.model
    
    def train(self, train_images, train_masks, val_images, val_masks):
        train_gen = DataGenerator(train_images, train_masks, self.config.BATCH_SIZE,
                                  (self.config.IMG_HEIGHT,self.config.IMG_WIDTH),
                                  self.config.NUM_CLASSES)
        val_gen = DataGenerator(val_images, val_masks, self.config.BATCH_SIZE,
                                (self.config.IMG_HEIGHT,self.config.IMG_WIDTH),
                                self.config.NUM_CLASSES, augment=False)
        callbacks = [
            keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7)
        ]
        self.model.fit(train_gen, validation_data=val_gen, epochs=self.config.EPOCHS, callbacks=callbacks, verbose=1)
    
    def save_model(self, filepath='landcover_segmentation_model.h5'):
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='landcover_segmentation_model.h5'):
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

def main_training_pipeline():
    config = LandCoverConfig()
    model = LandCoverSegmentationModel(config)
    print("Preparing dataset...")
    train_images, train_masks, val_images, val_masks = model.prepare_dataset()
    print("Building model...")
    model.build_and_compile_model()
    print("Training model...")
    model.train(train_images, train_masks, val_images, val_masks)
    model.save_model()
    print("Pipeline completed!")

if __name__ =="__main__":
    main_training_pipeline()