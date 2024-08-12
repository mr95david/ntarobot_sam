#!/usr/bin/env python3
import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from ament_index_python import get_package_share_directory

from ntarobot_sam2.sam2_client import SAMclient
from ntarobot_sam2.utils import show_anns
# from ros2_sam.sam_client import SAMClient
# from ros2_sam.utils import show_box, show_mask, show_points


def main(args: List[str] = None) -> None:
    # Inicializavcion de nodo
    rclpy.init(args=args)
    # Creacion de nodo
    sam_client = SAMclient(
        node_name = "sam2_client",
        service_name = "sam2_server/segment",
    )

    # Intento de ejecucion
    try:
        image = cv2.imread(
            os.path.join(get_package_share_directory("ntarobot_sam2"), "assets/cars.jpg")
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # points = np.array([[1035, 640], [1325, 610]])
        # labels = np.array([0, 0])
        # boxes = np.asarray([[54, 350, 1700, 1300]])

        masks = sam_client.sync_segment_request(
            image
        )

        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.show() 
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()