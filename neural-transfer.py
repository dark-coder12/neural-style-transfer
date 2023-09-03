import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import plotly.graph_objects as go
import plotly.io as pio

def loadImage(imagePath):
    
    maxDimension = 650
    img = tf.io.read_file(imagePath)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    longDimension = max(shape)
    scale = maxDimension / longDimension

    newShape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, newShape)
    img = img[tf.newaxis, :]
    return img

def tfToPILImage(tensor):
    tensor = tensor * 255

    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]

    return PIL.Image.fromarray(tensor)
    
def stylizeImages(contentImagePath, styleImagePath):
    contentImage = loadImage(contentImagePath)
    styleImage = loadImage(styleImagePath)

    hubModel = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    stylizedImage = hubModel(tf.constant(contentImage), tf.constant(styleImage))[0]

    return stylizedImage


def plotStylizedImages(contentImage, styleImage, finalImage):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker_opacity=0)) 

    fig.add_layout_image(
        source=contentImage,
        xref="x",
        yref="y",
        x=-0.1,
        y=0,
        sizex=0.35,
        sizey=0.35,
        xanchor="left",
        yanchor="top"
    )
    fig.add_layout_image(
        source=styleImage,
        xref="x",
        yref="y",
        x=0.25,
        y=0,
        sizex=0.4,
        sizey=0.4,
        xanchor="left",
        yanchor="top"
    )
    fig.add_layout_image(
        source=finalImage,
        xref="x",
        yref="y",
        x=0.7,
        y=0,
        sizex=0.4,
        sizey=0.4,
        xanchor="left",
        yanchor="top"
    )

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, range=[-0.1, 1]),  
        yaxis=dict(showgrid=False, zeroline=False, range=[-0.5, 0.1]),
        width=1720,  
        height=1000
    )
    return fig


def main():
    contentPath = 'izza.jpg'
    stylePath = 'starry_night.png'

    contentImage = PIL.Image.open(contentPath)
    styleImage = PIL.Image.open(stylePath)

    stylizedImage = stylizeImages(contentPath, stylePath)
    finalImage = tfToPILImage(stylizedImage)

    pio.write_html(plotStylizedImages(contentImage, styleImage, finalImage), 'output.html')
    finalImage.save("final-output.jpg")

if __name__ == "__main__":
    main()
