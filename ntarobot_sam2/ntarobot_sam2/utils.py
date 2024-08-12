# El siguiente programa realiza las funciones varias para la ejecucion del modelo de sam, 
# Teniendo en cuenta las funciones usadas en: https://github.com/ros-ai/ros2_sam/blob/main/ros2_sam/ros2_sam/utils.py
# Realizamos un procedimiento de traduccion para las nuevas funcionalidades de sam2

# Seccion de importe de librerias
# Seccion de librerias de procesamiento
import matplotlib.pyplot as plt
import numpy as np
# Librerias utilitarias

# No se esta incluyendo la validacion de existencia de la libreria de checkpoint

# Funcion para visualizar una seccion de la imagen marcada con un rectangulo
def show_box(box, ax, color: str = 'green', lw:int = 2) -> None:
    """ Ingresa las coordenadas de la caja, teniendo en cuenta los parametros:
    x0, y0: Coordenadas x y en la imagen
    w, h: Ancho y alto de la caja que se desea crear
    ax: Objeto figura a la que se le adicionara la caja"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor = color, facecolor=(0, 0, 0, 0), lw = lw)
    )

# Funcion que muestra los puntos de la segmentacion en la imagen
def show_points( coords, labels, ax, 
                marker_size = 375, color1:str = 'green',
                color2:str = 'red', marker_type = "*" ) -> None:
    """La siguiente funcion recibe como parametros de entrada:
    coords: Como los valores x y de la localizacion de puntos
    labels: Valores ingreasdos
    ax: Corresponde a la figura plt
    marker_size: El tamanho de los marcadores
    """
    # Puntos positivos
    pos_points = coords[labels == 1]
    # Puntos negativos
    neg_points = coords[labels == 0]
    # Visualizacion de marcadores en scatter 1
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color=color1,
        marker=marker_type,
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    # Visualizacion de marcadores en scatter 2
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color=color2,
        marker=marker_type,
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )

# Funcion para la visualizacion de mascara, es la que diferencia el objeto en la imagen ingresada
def show_mask(mask, ax, obj_id=None, random_color=False):
    """Parametros de entrada para la funcion
    mask: Matriz de valores de mascara de visualizacion
    ax: corresponde al objeto al que se le adicionara la mascara,
    obj_id: identificacion de objeto
    random_color: Validacion de color random para la asignacion del color de la mascara"""
    
    # condicional de cambio de color, color aleatorio o bien la asingacion de color especifico
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])

    # Valores de altura y anchura de la matriz de la mascara
    h, w = mask.shape[-2:]

    # Redimensionamiento de imagen, y asignacion de color
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    # Visualizaicon final de la imagem
    ax.imshow(mask_image)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    #sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    sorted_anns = anns
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0].shape[0], sorted_anns[0].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)
