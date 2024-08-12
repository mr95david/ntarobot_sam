# Seccion de importe de librerias
# Librerias de procesamiento
import cv2
import numpy as np
# Librerias de puente de comunicacion
from cv_bridge import CvBridge
# Librerias de ros
from rclpy.node import Node
# Librerias utilitarias
import os
# Librerias propias
from ntarobot_sam2.sam2 import SAM2
# Importe de ejecucion de servidor
from ntarobot_sam_msg.srv import Segmentacion, SegmentAuto

# Clase de ejecucion de nodo
class SAMserver(Node):
    def __init__(self, node_name:str = "sam2_server") -> None:
        super().__init__(node_name)
        # Mensaje de validacion de ejecucion de servidor
        self.get_logger().info("Inicio de ejecucion de servidor para procesamiento de imagenes")
        # Seccion de declaracion de parametros de ejecucion
        # Declracion multiple de parametros
        self.declare_parameters(
            namespace = "",
            parameters = [
                ("checkpoint_dir", ""), # Ruta de origen donde se localiza el nodo
                ("model_config_n", "sam2_hiera_s.yaml"), # Tipo de ejecucion de sam2
                ("decive", "cuda"), # Nombre de dispositivo de procesamiento
                ("val_auto", True)
            ]
        )

        # Obtencion de valores de parametros
        self._checkpoint_dir = str(self.get_parameter("checkpoint_dir").value)
        self.model_config_n = str(self.get_parameter("model_config_n").value)
        self.decive = str(self.get_parameter("decive").value)
        self.val_auto = bool(self.get_parameter("val_auto").value)

        # Inicializacion de variables de instancia
        self._init_vars()

        # Inicialziacion de servicios
        self._init_services()

        # Validacion de servicio listo para la ejecucion
        self.get_logger().info("Servicio listo para solicitudes")
    
    def _init_vars(self):
        # Variable de coneccion de puente
        self._bridge = CvBridge()
        # Inicializacion de ejecucion de sam
        if self.check_checkpoint_dir():
            self._sam = SAM2(
                self._checkpoint_dir,
                self.model_config_n,
                self.decive,
                self.val_auto
            )
            self.get_logger().info("Modelo cargado correctamente")
        else:
            raise ValueError("Error en la inicializacion del objeto de segmentacion")

    def _init_services(self) -> None:
        # Servicio de segmentacion
        self._sam2_segment_service = self.create_service(
            SegmentAuto, # Interfaz de segmentacion
            "~/segment",
            self.on_segment
        )

    # Funcion de ejecucion de servicio
    def on_segment(
        self, req: SegmentAuto.Request, res: SegmentAuto.Response
    ):
        self.get_logger().info("Solicitud de servicio recibida")

        # Intento de ejecucion de procesamiento
        try:
            # Primero se realiza la transformacion de la imagen desde el nodo a matriz de datos
            img_p = cv2.cvtColor(
                self._bridge.imgmsg_to_cv2(# Transformacion de imagen a csv
                    req.image
                ), cv2.COLOR_BGR2RGB # Conversion de imagen a valor en color
            )

            # En este caso no se esta realizando una seleccion de puntos
            # Tampoco se realiza la identificacion de cajas
            # Tampoco se agregan los labels

            # Se consigna el tiempo de inicio de ejecucion
            start_time = self.get_clock().now().nanoseconds

            # Se procesa la exploracion automatica de mascaras de la imagen ingresada
            mask_out = self._sam.mask_generator.generate(img_p)

            # Conteo de tiempo general de ejecucion
            final_time = round((self.get_clock().now().nanoseconds-start_time)/1.e9, 2)

            self.get_logger().info(f"El tiempo de ejecucion general fue {final_time}")

            # Validacion de tiempo de ejecucion de proceso
            res.masks = [self._bridge.cv2_to_imgmsg(m['segmentation'].astype(np.uint8)) for m in mask_out]

            # Salida general
            return res

        except Exception as e:
            raise ValueError(f"Error en la ejecucion. \n {e}")

    # Funcion para la validacion de existencia de la ruta de segmento
    def check_checkpoint_dir(self) -> bool:
        checkpoint_dir_default = "/home/eliodavid/test_sam/segment-anything-2/checkpoints/sam2_hiera_small.pt"
        if self._checkpoint_dir == "":
            self._checkpoint_dir = checkpoint_dir_default
        
        if os.path.exists(checkpoint_dir_default):
            return True
        self.get_logger().error("No existe la ruta de checkpoints")
        return False