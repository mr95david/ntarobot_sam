# Seccion de importe de librerias
# Importe librerias de procesamiento
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Importe de librerias utilitarias
import glob
import os
# Libreria para recoleccion de basura
import gc

# Creacion de clase de objeto para segmentacion de sam
class SAM2:
    # Se agregan los parametros de entrada por defecto para la creacion de la clase
    def __init__(
            self, checkpoint_dir:str, # Ruta de existencia de modelo
            model_config_n:str = "sam2_hiera_s.yaml", # Condiguracion determinada para el uso del modelo
            device: str = "cuda", # Tipo de dispositivo de ejecucion cuda/cpu
            val_auto: bool = True
            ) -> None:
        
        gc.collect()
        torch.cuda.empty_cache()
        # Inicializacion de dispositivos de gpu
        self._device = device
        self._init_device()

        # Inicializacion de predictor
        self.sam2 = build_sam2(
            model_config_n,
            checkpoint_dir,
            self._device,
            apply_postprocessing = False
        )

        # Validacion de automatico
        if val_auto:
            self.auto_segment()
    
    def _init_device(self) -> None:
        # Limpieza de cache de la gpu
        gc.collect()
        torch.cuda.empty_cache()

        # Configuracion de gpu y paquete de torch
        torch.autocast(
            device_type = self._device,
            dtype = torch.bfloat16
        ).__enter__()

        # Validacion de configuraciones necesarias de acuerdo al tamanho de ejecucion de la gpu
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def auto_segment(self):
        # Creacion de segmentacion automatica
        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2)
    
    # Funcion destructor de objeto
    def __del__(self):
        gc.collect()
        torch.cuda.empty_cache()

        
        