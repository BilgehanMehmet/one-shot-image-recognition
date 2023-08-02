import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import math
import tensorflow as tf 


def get_val_acc(train_imgs,test_imgs,labels,sess,nn,shape=None):
    val_acc = 0
    assert shape is not None, "Shape must be a tuple of (35,35), or (105,105)."
#     for _ in range(n_trials // __N_TRIALS_PER_ITER):           
    for index in range(train_imgs.shape[0]): #20-way 20 trials. Each iteration is one one-shot trial.
        current_image_train_set = np.asarray(list(train_imgs[index]) * 
            train_imgs.shape[0]).reshape((test_imgs.shape[0],) + shape + (1,))
        current_train_img_label = labels[0,index]
        probs = sess.run(nn.probabilities, feed_dict= {nn.X:current_image_train_set , nn.X2:test_imgs})
        predicted_class = np.argmin(probs)
        if labels[0,predicted_class] == labels[0,index]:
            val_acc += 1
        # print("probs:{}".format(probs))
        # print("predicted class:{}".format(predicted_class))
        # print("labels[predicted class]:{}, labels[true index]:{}".format(labels[0,predicted_class],labels[0,index]))
        # int("val_acc:{}".format(val_acc))
        # plot(train_imgs[predicted_class])
        # plot(test_imgs[index])
#         x = input("")
    return val_acc

def affine_transformation(char,seed,shift_range):
    assert(char.shape == (20,105,105,1))
    for idx,img in enumerate(char):
        char[idx] = perform_random_shift(img,seed=seed,shift_range=shift_range)
    return char



def plot(img):
    flag = False
    try:
        assert (img.shape == (105,105))
    except AssertionError as err:
        try:
            img = img.reshape(105,105)
        except:
            raise AssertionError("Image shape should be (105,105)")
    plt.imshow(img,cmap='gray')
    plt.show()
    

def transform_matrix_offset_center(transformation_matrix, width, height,seed):
        """ Corrects the offset of tranformation matrix
        
            Corrects the offset of tranformation matrix for the specified image 
            dimensions by considering the center of the image as the central point

            Args:
                transformation_matrix: transformation matrix from a specific
                    augmentation.
                width: image width
                height: image height

            Returns:
                The corrected transformation matrix.
        """
        np.random.seed(seed)
        o_x = float(width) / 2 + 0.5
        o_y = float(height) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transformation_matrix = np.dot(
            np.dot(offset_matrix, transformation_matrix), reset_matrix)

        return transformation_matrix
    

 # Applies a provided transformation to the image
def apply_transform(image, transformation_matrix,seed):
        """ Applies a provided transformation to the image

            Args:
                image: image to be augmented
                transformation_matrix: transformation matrix from a specific
                    augmentation.

            Returns:
                The transformed image
        """
        np.random.seed(seed)
        channel_axis = 2
        image = np.rollaxis(image, channel_axis, 0)
        final_affine_matrix = transformation_matrix[:2, :2]
        final_offset = transformation_matrix[:2, 2]

        channel_images = [ndi.interpolation.affine_transform(
            image_channel,
            final_affine_matrix,
            final_offset,
            order=0,
            mode='nearest',
            cval=0) for image_channel in image]

        image = np.stack(channel_images, axis=0)
        image = np.rollaxis(image, 0, channel_axis + 1)

        return image    


    
def perform_random_shift(image,seed,shift_range):
        """ Applies a random shift in x and y

            Args:
                image: image to be augmented
        
            Returns:
                The transformed image
        """ 
        np.random.seed(seed)
        tx = np.random.uniform(-shift_range[0],
                               shift_range[0])
        ty = np.random.uniform(-shift_range[1],
                               shift_range[1])

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        transformation_matrix = translation_matrix  # no need to do offset
        image = apply_transform(image, transformation_matrix,seed)

        return image


def perform_random_zoom(image,seed,zoom_range):
    """ Applies a random zoom
        Args:
            image: image to be augmented
    
        Returns:
            The transformed image
    """
    np.random.seed(seed)
    zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)

    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    transformation_matrix = transform_matrix_offset_center(
        zoom_matrix, image.shape[0], image.shape[1],seed)
    image = apply_transform(image, transformation_matrix,seed)

    return image


def perform_random_shear(image,seed,shear_range):
        """ Applies a random shear
            Args:
                image: image to be augmented
            Returns:
                The transformed image
        """
        np.random.seed(seed)
        shear = np.deg2rad(np.random.uniform(
            low=shear_range[0], high=shear_range[1]))

        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        transformation_matrix = transform_matrix_offset_center(
            shear_matrix, image.shape[0], image.shape[1],seed)
        image = apply_transform(image, transformation_matrix,seed)

        return image


def perform_random_rotation(image,seed,rotation_range):
        """ Applies a random rotation

            Args:
                image: image to be augmented

            Returns:
                The transformed image
        """
        np.random.seed(seed)
        theta = np.deg2rad(np.random.uniform(
            low=rotation_range[0], high=rotation_range[1]))

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])

        transformation_matrix = transform_matrix_offset_center(
            rotation_matrix, image.shape[0], image.shape[1],seed)
        image = apply_transform(image, transformation_matrix,seed)

        return image

def perform_affine_transformation(  image,
                                    seed,
                                    rotation_range,
                                    shear_range,
                                    shift_range,
                                    zoom_range):

    image = perform_random_rotation(image,seed,rotation_range)
    image = perform_random_shear(image,seed,shear_range)
    image = perform_random_shift(image,seed,shift_range)
    image = perform_random_zoom(image,seed,zoom_range)
    return image
