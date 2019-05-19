import argparse
import logging
import pickle

import imageio
import numpy as np
from scipy.ndimage.filters import convolve

# Displacements are by default saved to a file after every run. Once you have confirmed your
# LK code is working, you can load saved displacements to save time testing the
# rest of the project.
DEFAULT_DISPLACEMENTS_FILE = "final_displacements.pkl"


def bilinear_interp(image, points):
    """Given an image and an array of row/col (Y/X) points, perform bilinear
    interpolation and return the pixel values in the image at those points."""
    points = np.asarray(points)
    if points.ndim == 1:
        points = points[np.newaxis]

    valid = np.all(points < [image.shape[0] - 1, image.shape[1] - 1], axis=-1)
    valid *= np.all(points >= 0, axis=-1)
    valid = valid.astype(np.float32)
    points = np.minimum(points, [image.shape[0] - 2, image.shape[1] - 2])
    points = np.maximum(points, 0)

    fpart, ipart = np.modf(points)
    tl = ipart.astype(np.int32)
    br = tl + 1
    tr = np.concatenate([tl[..., 0:1], br[..., 1:2]], axis=-1)
    bl = np.concatenate([br[..., 0:1], tl[..., 1:2]], axis=-1)

    b = fpart[..., 0:1]
    a = fpart[..., 1:2]

    top = (1 - a) * image[tl[..., 0], tl[..., 1]] + \
          a * image[tr[..., 0], tr[..., 1]]
    bot = (1 - a) * image[bl[..., 0], bl[..., 1]] + \
          a * image[br[..., 0], br[..., 1]]
    return ((1 - b) * top + b * bot) * valid[..., np.newaxis]


def translate(image, displacement):
    """Takes an image and a displacement of the form X,Y and translates the
    image by the displacement. The shape of the output is the same as the
    input, with missing pixels filled in with zeros."""
    pts = np.mgrid[:image.shape[0], :image.shape[1]].transpose(1, 2, 0).astype(np.float32)
    pts -= displacement[::-1]

    return bilinear_interp(image, pts)


def convolve_img(image, kernel):
    """Convolves an image with a convolution kernel. Kernel should either have
    the same number of dimensions and channels (last dimension shape) as the
    image, or should have 1 less dimension than the image."""
    if kernel.ndim == image.ndim:
        if image.shape[-1] == kernel.shape[-1]:
            return np.dstack([convolve(image[..., c], kernel[..., c]) for c in range(kernel.shape[-1])])
        elif image.ndim == 2:
            return convolve(image, kernel)
        else:
            raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (
                kernel.shape, image.shape))
    elif kernel.ndim == image.ndim - 1:
        return np.dstack([convolve(image[..., c], kernel) for c in range(image.shape[-1])])
    else:
        raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (
            kernel.shape, image.shape))


def gaussian_kernel(ksize=5):
    """
    Computes a 2-d gaussian kernel of size ksize and returns it.
    """
    kernel = np.exp(-np.linspace(-(ksize // 2), ksize // 2, ksize)
                     ** 2 / 2) / np.sqrt(2 * np.pi)
    kernel = np.outer(kernel, kernel)
    kernel /= kernel.sum()
    return kernel


def lucas_kanade(H, I):
    """Given images H and I, compute the displacement that should be applied to
    H so that it aligns with I."""

    # Cylindrical warping introduces black pixels which should be ignored, and
    # motion in dark regions is difficult to estimate. Generate a binary mask
    # indicating pixels that are valid (average color value > 0.25) in both H
    # and I.
    mask = (H.mean(-1) > 0.25) * (I.mean(-1) > 0.25)
    mask = mask[:, :, np.newaxis]

    # Compute the partial image derivatives w.r.t. X, Y, and Time (t).
    # In other words, compute I_y, I_x, and I_t
    # To achieve this, use a _normalized_ 3x3 sobel kernel and the convolve_img
    # function above. NOTE: since you're convolving the kernel, you need to 
    # multiply it by -1 to get the proper direction.

    # divide by 8 to normalize
    sobel_x = -np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
    sobel_y = -np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0

    # convolve x and y dimension
    I_x = convolve_img(I, sobel_x)
    I_y = convolve_img(I, sobel_y)
    I_t = I - H

    # Compute the various products (Ixx, Ixy, Iyy, Ixt, Iyt) necessary to form
    # AtA. Apply the mask to each product.
    Ixx = (I_x * I_x) * mask
    Ixy = (I_x * I_y) * mask
    Iyy = (I_y * I_y) * mask
    Ixt = (I_x * I_t) * mask
    Iyt = (I_y * I_t) * mask

    # Build the AtA matrix and Atb vector. You can use the .sum() function on numpy arrays to help.
    AtA = np.array([[np.sum(Ixx), np.sum(Ixy)], [np.sum(Ixy), np.sum(Iyy)]])
    Atb = -np.array([np.sum(Ixt), np.sum(Iyt)])

    # Solve for the displacement using linalg.solve
    displacement = np.linalg.solve(AtA, Atb)

    # return the displacement and some intermediate data for unit testing..
    return displacement, AtA, Atb


def iterative_lucas_kanade(H, I, steps):
    # Run the basic Lucas Kanade algorithm in a loop `steps` times.
    # Start with an initial displacement of 0 and accumulate displacements.
    disp = np.zeros((2,), np.float32)
    for i in range(steps):
        # Translate the H image by the current displacement (using the translate function above)
        # H_translated = translate(H, disp)

        # run Lucas Kanade and update the displacement estimate
        # disp += lucas_kanade(H_translated, I)[0]

        # Condensed
        disp += lucas_kanade(translate(H, disp), I)[0]

    # Return the final displacement
    return disp


def gaussian_pyramid(image, levels):
    """
    Builds a Gaussian pyramid for an image with the given number of levels, then return it.
    Inputs:
        image: a numpy array (i.e., image) to make the pyramid from
        levels: how many levels to make in the gaussian pyramid
    Retuns:
        An array of images where each image is a blurred and shruken version of the first.
    """

    # Compute a gaussian kernel using the gaussian_kernel function above. You can leave the size as default.
    kernel = gaussian_kernel()

    # Add image to the the list as the first level
    pyr = [image]
    for level in range(1, levels):
        # Convolve the previous image with the gussian kernel
        # convolved_img = convolve_img(pyr[level - 1], kernel)

        # decimate the convolved image by downsampling the pixels in both dimensions.
        # Note: you can use numpy advanced indexing for this (i.e., ::2)
        # convolved_img = convolved_img[::2, ::2]

        # add the sampled image to the list
        # pyr.append(convolved_img)

        # Condensed
        pyr.append(convolve_img(pyr[level - 1], kernel)[::2, ::2])

    return pyr


def pyramid_lucas_kanade(H, I, initial_d, levels, steps):
    """Given images H and I, and an initial displacement that roughly aligns H
    to I when applied to H, run Iterative Lucas Kanade on a pyramid of the
    images with the given number of levels to compute the refined
    displacement."""

    initial_d = np.asarray(initial_d, dtype=np.float32)

    # Build Gaussian pyramids for the two images.
    # Flip em
    H_pyr = np.flip(gaussian_pyramid(H, levels))
    I_pyr = np.flip(gaussian_pyramid(I, levels))

    # Start with an initial displacement (scaled to the coarsest level of the
    # pyramid) and compute the updated displacement at each level using Lucas
    # Kanade.
    disp = initial_d / 2. ** (levels)
    for level in range(levels):
        # Get the two images for this pyramid level.
        H_level = H_pyr[level]
        I_level = I_pyr[level]

        # Scale the previous level's displacement and apply it to one of the
        # images via translation.
        disp *= 2.0

        H_translated = translate(H_level, disp)

        # Use the iterative Lucas Kanade method to compute a displacement
        # between the two images at this level.
        new_disp = iterative_lucas_kanade(H_translated, I_level, steps)

        # Update the displacement based on the one you just computed.
        disp += new_disp

    # Return the final displacement.
    return disp


def build_panorama(images, shape, displacements, initial_position, blend_width=16):
    # Allocate an empty floating-point image with space to store the panorama
    # with the given shape.
    image_height, image_width = images[0].shape[:2]
    pano_height, pano_width = shape
    panorama = np.zeros((pano_height, pano_width, 3), np.float32)

    # Place the last image, warped to align with the first, at its proper place
    # to initialize the panorama.
    cur_pos = initial_position
    cp = np.round(cur_pos).astype(np.int32)
    panorama[cp[0]: cp[0] + image_height, cp[1]: cp[1] +
                                                 image_width] = translate(images[-1], displacements[-1])

    # Place the images at their final positions inside the panorama, blending
    # each image with the panorama in progress. Use a blending window with the
    # given width.
    for i in range(len(images)):
        cp = np.round(cur_pos).astype(np.int32)

        overlap = image_width - abs(displacements[i][0])
        blend_start = int(overlap / 2 - blend_width / 2)
        blend_start_pano = int(cp[1] + blend_start)

        pano_region = panorama[cp[0]: cp[0] + image_height,
                      blend_start_pano: blend_start_pano + blend_width]
        new_region = images[i][:, blend_start: blend_start + blend_width]

        mask = np.zeros((image_height, blend_width, 1), np.float32)
        mask[:] = np.linspace(0, 1, blend_width)[np.newaxis, :, np.newaxis]
        mask[np.all(new_region == 0, axis=2)] = 0
        mask[np.all(pano_region == 0, axis=2)] = 1

        # blended_region = mask * new_region + (1 - mask) * pano_region
        blended_region = blend_with_mask(pano_region, new_region, mask)

        blended = images[i].copy("C")
        blended[:, blend_start: blend_start + blend_width] = blended_region
        blended[:, :blend_start] = panorama[cp[0]: cp[0] + image_height, cp[1]: blend_start_pano]

        panorama[cp[0]: cp[0] + blended.shape[0],
        cp[1]: cp[1] + blended.shape[1]] = blended
        cur_pos += -displacements[i][::-1]
        print("Placed %d." % i)

    # Return the finished panorama.
    return panorama


def mosaic(images, initial_displacements, load_displacements_from):
    """Given a list of N images taken in clockwise order and corresponding
    initial X/Y displacements of shape (N,2), refine the displacements and
    build a mosaic.

    initial_displacement[i] gives the translation that should be appiled to
    images[i] to align it with images[(i+1) % N]."""
    N = len(images)

    if load_displacements_from:
        print("Loading saved displacements...")
        final_displacements = pickle.load(open(load_displacements_from, "rb"))
    else:
        print("Refining displacements with Pyramid Iterative Lucas Kanade...")
        final_displacements = []
        for i in range(N):
            # Use Pyramid Iterative Lucas Kanade to compute displacements from
            # each image to the image that follows it, wrapping back around at
            # the end. A suggested number of levels and steps is 4 and 5
            # respectively. Make sure to append the displacement to
            # final_displacements so it gets saved to disk if desired.

            final_displacements.append(
                pyramid_lucas_kanade(images[i], images[(i + 1) % N], initial_displacements[i], 4, 5))

            # Some debugging output to help diagnose errors.
            print("Image %d:" % i,
                  initial_displacements[i], "->", final_displacements[i], "  ",
                  "%0.4f" % abs(
                      (images[i] - translate(images[(i + 1) % N], -initial_displacements[i]))).mean(), "->",
                  "%0.4f" % abs(
                      (images[i] - translate(images[(i + 1) % N], -final_displacements[i]))).mean()
                  )
        print('Saving displacements to ' + DEFAULT_DISPLACEMENTS_FILE)
        pickle.dump(final_displacements, open(DEFAULT_DISPLACEMENTS_FILE, "wb"))

    # Use the final displacements and the images' shape compute the full
    # panorama shape and the starting position for the first panorama image.
    sums = np.sum(final_displacements, axis=0)

    # Height = image_height + total vertical drift
    pano_height = np.int(images[0].shape[0] + sums[1])
    # Width = number of images * image_width - total horizontal drift
    pano_width = np.int(N * images[0].shape[1] + sums[0])
    # Initial position = [total_height - image_height, 0]
    initial_pos = np.array([pano_height - images[0].shape[0], 0], dtype=np.float)

    # Build the panorama.
    print("Building panorama...")
    panorama = build_panorama(images, (pano_height, pano_width), final_displacements, initial_pos.copy())
    return panorama, final_displacements


def warp_panorama(images, panorama, final_displacements):
    # Extra credit: Implement this function!

    # Resample the panorama image using a linear warp to distribute any vertical
    # drift to all of the sampling points. The final height of the panorama should
    # be equal to the height of one of the images.

    # Crop the panorama horizontally so that the left and right edges of the
    # panorama match (making it form a loop).

    # Return your final corrected panorama.
    warped = panorama
    return warped


def blend_with_mask(source, target, mask):
    """
    Blends the source image with the target image according to the mask.
    Pixels with value "1" are source pixels, "0" are target pixels, and
    intermediate values are interpolated linearly between the two.
    Args:
        source:     The source image.
        target:     The target image.
        mask:       The mask to use
    Returns:
        A new image representing the linear combination of the mask (and it's inverse)
        with source and target, respectively.
    """

    # First we need to define some functions
    def split(img):
        """
        Splits a image (a height x width x 3 ndarray) into 3 ndarrays, 1 for each channel.
        Args:
            img:    A height x width x 3 channel ndarray.
        Returns:
            A 3-tuple of the r, g, and b channels.
        """
        if img.shape[2] != 3:
            raise ValueError('The split function requires a 3-channel input image')

        return img[:, :, 0], img[:, :, 1], img[:, :, 2]

    def merge(r, g, b):
        """
        Merges three images (height x width ndarrays) into a 3-channel color image ndarrays.
        Args:
            r:    A height x width ndarray of red pixel values.
            g:    A height x width ndarray of green pixel values.
            b:    A height x width ndarray of blue pixel values.
        Returns:
            A height x width x 3 ndarray representing the color image.
        """

        return np.dstack((r, g, b))

    def expand(image):
        # upsample to be twice the size of the input image
        up = np.zeros(shape=(2 * image.shape[0], 2 * image.shape[1]))

        # take every other row/column from input
        up[::2, ::2] = image

        # convolve that thang
        kernel = gaussian_kernel()
        convolution = convolve_img(up, kernel)

        # scale back up and return
        return convolution * 4

    def laplacian_pyramid(gaussian_pyramid):
        pyramid = []

        for i in range(len(gaussian_pyramid) - 1):
            expanded = expand(gaussian_pyramid[i + 1])

            # might need to crop it
            if len(expanded) != len(gaussian_pyramid[i]):
                expanded = expanded[0:len(gaussian_pyramid[i]), :]

            if len(expanded[0]) != len(gaussian_pyramid[i][0]):
                expanded = expanded[:, 0:len(gaussian_pyramid[i][0])]

            pyramid.append(gaussian_pyramid[i] - expanded)

        pyramid.append(gaussian_pyramid.pop())

        return pyramid

    def blend(white_pyramid, black_pyramid, mask_pyramid):
        blended_pyramid = []

        for i in range(len(mask_pyramid)):
            blended_pyramid.append(mask_pyramid[i] * white_pyramid[i] + (1 - mask_pyramid[i]) * black_pyramid[i])

        return blended_pyramid

    def collapse(pyramid):
        output = pyramid[-1]

        # collapse the pyramid
        for i in range(len(pyramid) - 1, 0, -1):
            expanded = expand(output)

            # might need to crop it
            if len(expanded) != len(pyramid[i - 1]):
                expanded = expanded[0:len(pyramid[i - 1]), :]

            if len(expanded[0]) != len(pyramid[i - 1][0]):
                expanded = expanded[:, 0:len(pyramid[i - 1][0])]

            output = expanded + pyramid[i - 1]

        # Convert the result to be the same type as source and return the result
        return output

    # split images into channels
    (source_r, source_g, source_b) = split(source)
    (target_r, target_g, target_b) = split(target)
    (mask_r, mask_g, mask_b) = split(np.dstack((mask, mask, mask)))

    source_r, source_g, source_b = source_r.astype(np.float), source_g.astype(np.float), source_b.astype(np.float)
    target_r, target_g, target_b = target_r.astype(np.float), target_g.astype(np.float), target_b.astype(np.float)

    mask_r = mask_r.astype(np.float) / 255.0
    mask_g = mask_g.astype(np.float) / 255.0
    mask_b = mask_b.astype(np.float) / 255.0

    gaussian_source_r = gaussian_pyramid(source_r, 6)
    gaussian_source_g = gaussian_pyramid(source_g, 6)
    gaussian_source_b = gaussian_pyramid(source_b, 6)

    gaussian_target_r = gaussian_pyramid(target_r, 6)
    gaussian_target_g = gaussian_pyramid(target_g, 6)
    gaussian_target_b = gaussian_pyramid(target_b, 6)

    gaussian_mask_r = gaussian_pyramid(mask_r, 6)
    gaussian_mask_g = gaussian_pyramid(mask_g, 6)
    gaussian_mask_b = gaussian_pyramid(mask_b, 6)

    laplacian_source_r = laplacian_pyramid(gaussian_source_r)
    laplacian_source_g = laplacian_pyramid(gaussian_source_g)
    laplacian_source_b = laplacian_pyramid(gaussian_source_b)

    laplacian_target_r = laplacian_pyramid(gaussian_target_r)
    laplacian_target_g = laplacian_pyramid(gaussian_target_g)
    laplacian_target_b = laplacian_pyramid(gaussian_target_b)

    blended_r = blend(laplacian_source_r, laplacian_target_r, gaussian_mask_r)
    blended_g = blend(laplacian_source_g, laplacian_target_g, gaussian_mask_g)
    blended_b = blend(laplacian_source_b, laplacian_target_b, gaussian_mask_b)

    output_r = collapse(blended_r)
    output_g = collapse(blended_g)
    output_b = collapse(blended_b)

    output_r[output_r < 0] = 0
    output_r[output_r > 255] = 255

    output_g[output_g < 0] = 0
    output_g[output_g > 255] = 255

    output_b[output_b < 0] = 0
    output_b[output_b > 255] = 255

    return merge(output_r, output_g, output_b)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Creates a mosaic by stitching together a provided set of images.')
    parser.add_argument(
        'input', type=str, help='A txt file containing the images and initial displacement positions.')
    parser.add_argument('output', type=str,
                        help='What image file to save the panorama to.')
    parser.add_argument('--displacements', type=str,
                        help='Load displacements from this pickle file (useful for build_panorama).', default=None)
    args = parser.parse_args()

    filenames, xinit, yinit = zip(
        *[l.strip().split() for l in open(args.input).readlines()])
    xinit = np.array([float(x) for x in xinit])[:, np.newaxis]
    yinit = np.array([float(y) for y in yinit])[:, np.newaxis]
    disps = np.hstack([xinit, yinit])

    images = [imageio.imread(fn)[:, :, :3].astype(
        np.float32) / 255. for fn in filenames]

    panorama, final_displacements = mosaic(images, disps, args.displacements)

    result = warp_panorama(images, panorama, final_displacements)
    imageio.imwrite(args.output, result)
