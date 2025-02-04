"""
Visualization Utils
Adapted/copied from https://gist.github.com/WChargin/d8eb0cbafc4d4479d004#file-transforms-py
"""

import numpy as np
import PIL.Image


class RGBTransform(object):
    """A description of an affine transformation to an RGB image.
    This class is immutable.
    Methods correspond to matrix left-multiplication/post-application:
    for example,
        RGBTransform().multiply_with(some_color).desaturate()
    describes a transformation where the multiplication takes place first.
    Use rgbt.applied_to(image) to return a converted copy of the given image.
    For example:
        grayish = RGBTransform.desaturate(factor=0.5).applied_to(some_image)
    """

    def __init__(self, matrix=None):
        self._matrix = matrix if matrix is not None else np.eye(4)

    def _then(self, operation):
        return RGBTransform(np.dot(_embed44(operation), self._matrix))

    def desaturate(self, factor=1.0, weights=(0.299, 0.587, 0.114)):
        """Desaturate an image by the given amount.
        A factor of 1.0 will make the image completely gray;
        a factor of 0.0 will leave the image unchanged.
        The weights represent the relative contributions of each channel.
        They should be a 1-by-3 array-like object (tuple, list, np.array).
        In most cases, their values should sum to 1.0
        (otherwise, the transformation will cause the image
        to get lighter or darker).
        """
        weights = _to_rgb(weights, "weights")

        # tile: [wr, wg, wb]  ==>  [[wr, wg, wb], [wr, wg, wb], [wr, wg, wb]]
        desaturated_component = factor * np.tile(weights, (3, 1))
        saturated_component = (1 - factor) * np.eye(3)
        operation = desaturated_component + saturated_component

        return self._then(operation)

    def multiply_with(self, base_color, factor=1.0):
        """Multiply an image by a constant base color.
        The base color should be a 1-by-3 array-like object
        representing an RGB color in [0, 255]^3 space.
        For example, to multiply with orange,
        the transformation
            RGBTransform().multiply_with((255, 127, 0))
        might be used.
        The factor controls the strength of the multiplication.
        A factor of 1.0 represents straight multiplication;
        other values will be linearly interpolated between
        the identity (0.0) and the straight multiplication (1.0).
        """
        component_vector = _to_rgb(base_color, "base_color") / 255.0
        new_component = factor * np.diag(component_vector)
        old_component = (1 - factor) * np.eye(3)
        operation = new_component + old_component

        return self._then(operation)

    def mix_with(self, base_color, factor=1.0):
        """Mix an image by a constant base color.
        The base color should be a 1-by-3 array-like object
        representing an RGB color in [0, 255]^3 space.
        For example, to mix with orange,
        the transformation
            RGBTransform().mix_with((255, 127, 0))
        might be used.
        The factor controls the strength of the color to be added.
        If the factor is 1.0, all pixels will be exactly the new color;
        if it is 0.0, the pixels will be unchanged.
        """
        base_color = _to_rgb(base_color, "base_color")
        operation = _embed44((1 - factor) * np.eye(3))
        operation[:3, 3] = factor * base_color

        return self._then(operation)

    def get_matrix(self):
        """Get the underlying 3-by-4 matrix for this affine transform."""
        return self._matrix[:3, :]

    def applied_to(self, image):
        """Apply this transformation to a copy of the given RGB* image.
        The image should be a PIL image with at least three channels.
        Specifically, the RGB and RGBA modes are both supported, but L is not.
        Any channels past the first three will pass through unchanged.
        The original image will not be modified;
        a new image of the same mode and dimensions will be returned.
        """

        # PIL.Image.convert wants the matrix as a flattened 12-tuple.
        # (The docs claim that they want a 16-tuple, but this is wrong;
        # cf. _imaging.c:767 in the PIL 1.1.7 source.)
        matrix = tuple(self.get_matrix().flatten())

        channel_names = image.getbands()
        channel_count = len(channel_names)
        if channel_count < 3:
            raise ValueError("Image must have at least three channels!")
        elif channel_count == 3:
            return image.convert("RGB", matrix)
        else:
            # Probably an RGBA image.
            # Operate on the first three channels (assuming RGB),
            # and tack any others back on at the end.
            channels = list(image.split())
            rgb = PIL.Image.merge("RGB", channels[:3])
            transformed = rgb.convert("RGB", matrix)
            new_channels = transformed.split()
            channels[:3] = new_channels
            return PIL.Image.merge("".join(channel_names), channels)

    def applied_to_pixel(self, color):
        """Apply this transformation to a single RGB* pixel.
        In general, you want to apply a transformation to an entire image.
        But in the special case where you know that the image is all one color,
        you can save cycles by just applying the transformation to that color
        and then constructing an image of the desired size.
        For example, in the result of the following code,
        image1 and image2 should be identical:
            rgbt = create_some_rgb_tranform()
            white = (255, 255, 255)
            size = (100, 100)
            image1 = rgbt.applied_to(PIL.Image.new("RGB", size, white))
            image2 = PIL.Image.new("RGB", size, rgbt.applied_to_pixel(white))
        The construction of image2 will be faster for two reasons:
        first, only one PIL image is created; and
        second, the transformation is only applied once.
        The input must have at least three channels;
        the first three channels will be interpreted as RGB,
        and any other channels will pass through unchanged.
        To match the behavior of PIL,
        the values of the resulting pixel will be rounded (not truncated!)
        to the nearest whole number.
        """
        color = tuple(color)
        channel_count = len(color)
        extra_channels = tuple()
        if channel_count < 3:
            raise ValueError("Pixel must have at least three channels!")
        elif channel_count > 3:
            color, extra_channels = color[:3], color[3:]

        color_vector = np.array(color + (1,)).reshape(4, 1)
        result_vector = np.dot(self._matrix, color_vector)
        result = result_vector.flatten()[:3]

        full_result = tuple(result) + extra_channels
        rounded = tuple(int(round(x)) for x in full_result)

        return rounded


def _embed44(matrix):
    """Embed a 4-by-4 or smaller matrix in the upper-left of I_4."""
    result = np.eye(4)
    r, c = matrix.shape
    result[:r, :c] = matrix
    return result


def _to_rgb(thing, name="input"):
    """Convert an array-like object to a 1-by-3 numpy array, or fail."""
    thing = np.array(thing)
    assert thing.shape == (
        3,
    ), "Expected %r to be a length-3 array-like object, but found shape %s" % (
        name,
        thing.shape,
    )
    return thing


def apply_filter(img, color=(0, 255, 0), factor=0.15):
    img_proc = PIL.Image.fromarray(img)
    img_proc = RGBTransform().mix_with(color, factor=factor).applied_to(img_proc)
    img_proc = np.array(img_proc)
    return img_proc
