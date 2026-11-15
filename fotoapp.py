import os
import requests
from io import BytesIO
import numpy as np  
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter

class ImagenNoCargadaError(Exception):
    """Excepción personalizada lanzada cuando se intenta editar sin cargar imagen previa (Punto 5)."""
    pass


# 1. FUNCIÓN DE CARGA Y REDIMENSIONADO SIN DISTORSIÓN
def cargar_y_redimensionar(ruta_o_url: str, red_social: str) -> Image.Image:
    """
    Lee una foto (local o URL) y la redimensiona al formato de la red social
    especificada manteniendo la relación de aspecto original (Punto 1).
    """
    dimensiones = {
        "youtube": (1280, 720),
        "instagram": (1080, 1080),
        "twitter": (1200, 675),
        "facebook": (1200, 630)
    }
    
    clave = red_social.strip().lower()
    if clave not in dimensiones:
        raise ValueError(f"Red social no válida. Opciones permitidas: {list(dimensiones.keys())}")
    
    if ruta_o_url.startswith(("http://", "https://")):
        headers = {"User-Agent": "Mozilla/5.0"}
        respuesta = requests.get(ruta_o_url, headers=headers, timeout=20)
        respuesta.raise_for_status()
        img = Image.open(BytesIO(respuesta.content))
    else:
        if not os.path.exists(ruta_o_url):
            raise FileNotFoundError(f"No se encontró el archivo en la ruta: {ruta_o_url}")
        img = Image.open(ruta_o_url)

    # Convertir a RGB para evitar problemas con canales alfa (PNG transparentes)
    img = img.convert("RGB")

    # ImageOps.pad ajusta la imagen al tamaño sin distorsionarla y rellena los espacios vacíos
    target_size = dimensiones[clave]
    img_redimensionada = ImageOps.pad(img, target_size, color=(0, 0, 0))
    
    return img_redimensionada


# 2. FUNCIÓN DE AJUSTE DE CONTRASTE MEDIANTE HISTOGRAMA
def ajustar_contraste_histograma(img: Image.Image, nombre_salida: str = "contraste_resultado.png") -> Image.Image:
    """
    Ecualiza el histograma de la imagen para mejorar el contraste,
    muestra la comparativa visual y guarda la figura (Punto 2).
    """
    if img is None:
        raise ImagenNoCargadaError("Primero debes cargar una imagen usando la Opción 1.")
    
    img_ecualizada = ImageOps.equalize(img)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title("Imagen Original")
    axes[0].axis("off")
    
    axes[1].imshow(img_ecualizada)
    axes[1].set_title("Contraste Ajustado (Ecualización)")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(nombre_salida, dpi=150)
    plt.show()
    
    return img_ecualizada


# 3. FUNCIÓN PARA APLICAR LOS 9 FILTROS DE PILLOW
def aplicar_filtro_pillow(img: Image.Image, opcion_filtro: int, carpeta_salida: str = "resultados_filtros") -> Image.Image:
    """
    Aplica uno de los 9 filtros de Pillow elegidos por el usuario,
    muestra la imagen original al lado de la filtrada y guarda la figura (Punto 3).
    """
    if img is None:
        raise ImagenNoCargadaError("Primero debes cargar una imagen usando la Opción 1.")

    filtros = {
        1: ("Blur", ImageFilter.BLUR),
        2: ("Contour", ImageFilter.CONTOUR),
        3: ("Detail", ImageFilter.DETAIL),
        4: ("Edge Enhance", ImageFilter.EDGE_ENHANCE),
        5: ("Edge Enhance More", ImageFilter.EDGE_ENHANCE_MORE),
        6: ("Emboss", ImageFilter.EMBOSS),
        7: ("Find Edges", ImageFilter.FIND_EDGES),
        8: ("Sharpen", ImageFilter.SHARPEN),
        9: ("Smooth", ImageFilter.SMOOTH),
    }

    if opcion_filtro not in filtros:
        raise ValueError("El número de filtro debe estar entre 1 y 9.")

    nombre_filtro, filtro_obj = filtros[opcion_filtro]
    img_filtrada = img.filter(filtro_obj)

    os.makedirs(carpeta_salida, exist_ok=True)
    ruta_guardado = os.path.join(carpeta_salida, f"filtro_{opcion_filtro}_{nombre_filtro.replace(' ', '_')}.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(img_filtrada)
    axes[1].set_title(f"Filtro: {nombre_filtro}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(ruta_guardado, dpi=150)
    plt.show()

    return img_filtrada


# 4. FUNCIÓN DE BOCETO / DETECCIÓN DE BORDES (Umbralización vs Canny)
def generar_boceto_bordes(img: Image.Image, umbral_binario: int = 127, canny_min: int = 80, canny_max: int = 180, nombre_salida: str = "boceto_comparativa.png"):
    """
    Aplica dos técnicas distintas (Umbralización y Canny) para resaltar contornos
    tipo boceto y muestra los resultados lado a lado (Punto 4).
    """
    if img is None:
        raise ImagenNoCargadaError("Primero debes cargar una imagen usando la Opción 1.")

    # Convertir imagen PIL a formato OpenCV (escala de grises)
    img_np = np.array(img)
    gris = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gris_suave = cv2.GaussianBlur(gris, (5, 5), 0)

    # Técnica 1: Umbralización binaria
    _, bordes_umbral = cv2.threshold(gris_suave, umbral_binario, 255, cv2.THRESH_BINARY)

    # Técnica 2: Detector de Canny (invertido para aspecto de boceto a lápiz)
    bordes_canny = cv2.Canny(gris_suave, canny_min, canny_max)
    boceto_canny = cv2.bitwise_not(bordes_canny)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(bordes_umbral, cmap="gray")
    axes[1].set_title(f"Umbralización (T={umbral_binario})")
    axes[1].axis("off")

    axes[2].imshow(boceto_canny, cmap="gray")
    axes[2].set_title(f"Boceto Canny ({canny_min}, {canny_max})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(nombre_salida, dpi=150)
    plt.show()

    return bordes_umbral, boceto_canny

# 5. MENÚ INTERACTIVO CON VALIDACIÓN DE FLUJO
def ejecutar_menu():
    imagen_actual = None

    while True:
        print("\n================ FOTOAPP - MENÚ PRINCIPAL ================")
        print("1. Cargar y redimensionar imagen (Ej: imagenes/messi.jpg)")
        print("2. Ajustar contraste (Ecualización de histograma)")
        print("3. Aplicar filtros de Pillow (1 al 9)")
        print("4. Generar boceto / detección de bordes (Umbralización vs Canny)")
        print("5. Salir")
        
        opcion = input("Selecciona una opción (1-5): ").strip()

        try:
            if opcion == "1":
                ruta = input("Ingresa la ruta de la imagen [Por defecto: imagenes/messi.jpg]: ").strip()
                if not ruta:
                    ruta = "imagenes/messi.jpg"
                red = input("Red social (youtube, instagram, twitter, facebook) [Por defecto: instagram]: ").strip()
                if not red:
                    red = "instagram"
                
                imagen_actual = cargar_y_redimensionar(ruta, red)
                print(f"¡Imagen cargada y adaptada con éxito! Dimensiones: {imagen_actual.size}")

            elif opcion == "2":
                ajustar_contraste_histograma(imagen_actual)

            elif opcion == "3":
                print("\nFiltros disponibles:")
                print(" 1. Blur | 2. Contour | 3. Detail | 4. Edge Enhance | 5. Edge Enhance More")
                print(" 6. Emboss | 7. Find Edges | 8. Sharpen | 9. Smooth")
                num = int(input("Elige el número de filtro (1-9): "))
                aplicar_filtro_pillow(imagen_actual, num)

            elif opcion == "4":
                generar_boceto_bordes(imagen_actual)

            elif opcion == "5":
                print("Saliendo de FotoApp. ¡Mucho éxito con tu defensa!")
                break
            else:
                print("Opción inválida. Por favor ingresa un número del 1 al 5.")

        except ImagenNoCargadaError as e:
            print(f"\n[ALERTA DE FLUJO]: {e}")
        except (ValueError, FileNotFoundError, requests.RequestException) as e:
            print(f"\n[ERROR DE DATOS]: {e}")
        except Exception as e:
            print(f"\n[ERROR INESPERADO]: {e}")


if __name__ == "__main__":
    print("Iniciando FotoApp...")
    # Prueba automática rápida para verificar que tu imagen de Messi se cargue bien al iniciar
    if os.path.exists("imagenes/messi.jpg"):
        print("✓ Encontramos 'imagenes/messi.jpg'. ¿Quieres probar la carga automática? (s/n): ", end="")
        if input().strip().lower() == 's':
            try:
                img_prueba = cargar_y_redimensionar("imagenes/messi.jpg", "instagram")
                print(f"¡Carga de prueba exitosa! Tamaño resultante: {img_prueba.size}")
            except Exception as e:
                print(f"Error en la prueba automática: {e}")
    
    # Arrancamos el menú interactivo completo
    ejecutar_menu()