#!/usr/bin/env python3
# Librerias de ros2
import rclpy
# Importe de funcion de ejecucion de nodo desde el paquete sam2_server
from ntarobot_sam2.sam2_server import SAMserver
# importe de tipo de dato para ejecucion
from typing import List

def main(args: List[str] = None) -> None:
    # Inicializacion de nodos con argumentos
    rclpy.init(args=args)

    # Asignacion de nodo
    sam_server = SAMserver()
     # Cierre de nodos
    try:
        rclpy.spin(sam_server)
    except KeyboardInterrupt:
        pass
    finally:
        sam_server.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == "__main__":
    main()