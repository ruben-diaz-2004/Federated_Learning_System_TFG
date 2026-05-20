"""
@author: José Ignacio Estévez Damas
@author: Rubén Díaz Marrero
Grado en ingeniería informática, Universidad de La Laguna
Trabajo de Fin de Grado — Curso 2025/2026
======================

data_preprocessing.py

Preprocesamiento de datos con Hugging Face Datasets y torchvision.transforms.
"""
import sys
import os
from pathlib import Path
from datasets import load_dataset,DownloadMode,ReadInstruction

from torchvision.transforms.v2 import Resize,InterpolationMode
import matplotlib.pyplot as plt

class Data_Preprocessing:
    def __init__(
            self,
            data_path=Path("./data/"), #Ruta a un directorio de imagenes con dos clases
            split_name='train',
            image_size=[256,256], # Tamaño de la imagen.
            image_processor=None, # Procesamiento de batches de imagenes justo antes de ser consumidas por el modelo
            num_proc=1, # workers
            prep_batch_size=32, # Tam. batches
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS, # Dataset download mode
            keep_in_memory=True # Opcion para el Dataset

    ):
        self.data_path=data_path
        self.split_name=split_name
        self.image_size=image_size
        if image_processor is None:
            self.image_processor=self._default_image_processor
        self.num_proc=num_proc
        self.batch_size=prep_batch_size
        self.download_mode=download_mode
        self.keep_in_memory=keep_in_memory
        self._build_dataset()
        # Transformacion que solo se aplica una vez
        self.dataset=self.dataset.map(self._prepare,
                                      batched=True,
                                      batch_size=self.batch_size,
                                      num_proc=self.num_proc)
        self.dataset.set_transform(lambda ex : self._transform(ex,None))

    def show_sample(self,sample,show_image=True):

        image=self.dataset[sample]["proc_image"]
        label=self.dataset[sample]["label"]
        name=self.dataset[sample]["name"]
        print(f"Name:{name} Label:{label}")
        if show_image:
            plt.imshow(image) # imshow espera fila, columna, canal. De los datos viene como: canal, columna, fila.


    def _build_dataset(self):
        # Usamos directamente el nombre del split que pasamos al instanciar
        self.dataset = load_dataset("imagefolder",
                data_dir=self.data_path,
                split=self.split_name, 
                download_mode=self.download_mode,
                keep_in_memory=False) # keep_in_memory False para reutilizar el dataset del cache


    def _prepare(self,examples):
        # _prepare se aplica una vez a todo el dataset
        #print(examples)
        transform=Resize(self.image_size,
                         interpolation=InterpolationMode.BILINEAR)
        examples["image"] = [transform(im.convert("RGB")) for im in examples["image"]]
        return examples

    def _transform(self, examples, _trans=None):
        # _transform se aplica cada vez que se usa el dataset para instanciar imagenes.
        if "image" in examples.keys():
            if _trans is not None:
                examples["proc_image"]=[_trans(img.convert("RGB")) for img in examples["image"]]
            else:
                examples["proc_image"]=[ img.convert("RGB") for img in examples["image"]]
            
            procesado = self.image_processor(examples["proc_image"])

            if isinstance(procesado, dict) and "pixel_values" in procesado:
                examples["pixel_values"] = procesado["pixel_values"]
            else:
                examples["pixel_values"] = procesado

        return examples

    def _default_image_processor(self,examples):
        return examples

    def len(self):
        return len(self.dataset)
