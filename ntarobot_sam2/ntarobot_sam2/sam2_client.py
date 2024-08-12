# Seccion de importe de librerias 
# Librerias de ros
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
# Librerias de interfaces
from std_msgs.msg import Int32MultiArray
# Librerias para proseamiento y manejo de valores 
from typing import List, Optional, Tuple
import numpy as np
# Importe de mensaje de servicio
from ntarobot_sam_msg.srv import SegmentAuto

# Creacion de nodo de ejecucion
class SAMclient(Node):
    def __init__(
        self, node_name: str = "sam2_client",
        service_name: str = "sam2_server/segment"
    ) -> None:
        super().__init__(node_name)

        # Inicializacion de variables
        self._init_vars()
        # Inicializacion de cliente
        self._init_client(service_name)

        # Procesamiento de ejecucion de cliente
        while not self.sam_segment_client.wait_for_service(
            timeout_sec = 1.0 # Tiempo de espera para ejecucion de procesamiento del servicio solicitado
        ):
            # Validacion de ejecucion actual de nodo
            if not rclpy.ok():
                # Validacion de ejecucion
                self.get_logger().error("Ejecucion de cliente interrumpida.")
                return
            
            self.get_logger().info(
                f"Esperando por la ejecucion del servicio '{service_name}'..."
            )

        # Validacion de ejecucion
        self.get_logger().info(f"Establecido el cliente para el servicio '{service_name}'.")

    # Funcion de inicializacion de variables de instancia
    def _init_vars(self) -> None:
        self._bridge = CvBridge()
    
    # funcion de inicializacion de cliente de ejecucion
    def _init_client(self, service_name) -> None:
        # creacion de cliente de servicio especifico
        self.sam_segment_client = self.create_client(
            SegmentAuto,
            f"{service_name}"
        )

    # funcion de procesamiento de imagen para segmentacion
    def sync_segment_request(
            self,
            img_rgb: np.ndarray, # Imagen de entrada en formato a color
    ):
        # Ejecucion de valor de solicitud de procesamiento
        future = self.sam_segment_client.call_async(
            SegmentAuto.Request(
                image = self._bridge.cv2_to_imgmsg(img_rgb)
            )
        )

        # Ejecucion hasta completar la solicitud
        rclpy.spin_until_future_complete(self, future = future)

        # Validacion de retorno nulo
        if future.result() is None:
            raise RuntimeError("Error en la solicitud de segmentacion")
        
        # asignacion de valor de salida
        res = future.result()

        # Retorno de valor obtenido
        return [self._bridge.imgmsg_to_cv2(m) for m in res.masks]