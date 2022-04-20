import numpy as np
import cv2
from PIL import Image
import random
import torch
import torch.nn as nn
from scipy import ndimage, misc
from scipy.spatial.distance import cdist
# import pandas as pd

class Masks:

    @staticmethod
    def get_ff_mask(h, w, num_v = None):
        #Source: Generative Inpainting https://github.com/JiahuiYu/generative_inpainting

        mask = np.zeros((h,w))
        if num_v is None:
            num_v = 15+np.random.randint(9) #5
        SEGMENT_LENGTH = int(min(h,w)*0.3)
        BRUSH_WIDTH = int(min(h,w)*0.05)
        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            for j in range(1+np.random.randint(5)):
                angle = 0.01+np.random.randint(4.0)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10+np.random.randint(SEGMENT_LENGTH) # 40
                brush_w = BRUSH_WIDTH + np.random.randint(BRUSH_WIDTH) # 10
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y

        return mask.astype(np.float32)


    @staticmethod
    def get_box_mask(h,w):
        height, width = h, w

        mask = np.zeros((height, width))

        mask_width = random.randint(int(0.3 * width), int(0.7 * width)) 
        mask_height = random.randint(int(0.3 * height), int(0.7 * height))
 
        mask_x = random.randint(0, width - mask_width)
        mask_y = random.randint(0, height - mask_height)

        mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
        return mask

    @staticmethod
    def get_ca_mask(h,w, scale = None, r = None):

        if scale is None:
            scale = random.choice([1,2,4,8])
        if r is None:
            r = random.randint(2,6) # repeat median filter r times

        height = h
        width = w
        mask = np.random.randint(2, size = (height//scale, width//scale))

        for _ in range(r):
            mask = ndimage.median_filter(mask, size=3, mode='constant')
        mask = cv2.resize(mask,(w,h), interpolation=cv2.INTER_NEAREST)
        # mask = transform.resize(mask, (h,w)) # misc.imresize(mask,(h,w),interp='nearest')
        if scale > 1:
            struct = ndimage.generate_binary_structure(2, 1)
            mask = ndimage.morphology.binary_dilation(mask, struct)


        return mask

    @staticmethod
    def get_random_mask(h,w):
        f = random.choice([Masks.get_box_mask, Masks.get_ca_mask, Masks.get_ff_mask]) 
        return f(h,w).astype(np.int32)


def bounding_box(mask):
    '''
    Finds bounding box (x_left,x_right, y_top,y_bottom) of mask
    '''
    a = np.where(mask != 0)
    if a[0].size==0 or a[1].size==0:
        return None
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    if bbox[1]-bbox[0] <=1 or bbox[3] - bbox[2] <= 1:
        return None
    return bbox


def resize_to_box(img, to_size=(224,224),keep_aspect=True,
    bg_color=(0,0,0), interpolation=cv2.INTER_LINEAR):
    '''
    resize to to_size box and fill with bg_color
    '''
    old_size = img.shape[:2]
    ratio = min(to_size[0]/ old_size[0], to_size[1]/ old_size[1])
    if keep_aspect:
        new_size = tuple([round(x * ratio) for x in old_size])
    else:
        new_size = to_size
    if ratio < 0.2:
        img = cv2.blur(img,(3,3))
    resized_img = cv2.resize(img.astype(np.uint8),(new_size[1],new_size[0]),interpolation=interpolation)
    res_image = np.ones((*to_size,len(bg_color))) * bg_color
    res_image[(to_size[0]-new_size[0])//2:(to_size[0]-new_size[0])//2+new_size[0],
        (to_size[1]-new_size[1])//2:(to_size[1]-new_size[1])//2+new_size[1],:] = resized_img
    return res_image.astype(np.uint8)


def denormalize_tensor(t:torch.Tensor):
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])
    x = t * STD[:, None, None] + MEAN[:, None, None]
    return x

class FreeFormMask(object):
    """
    Free form masking
    args: nodes - number of masking nodes
    """

    def __init__(self, nodes = 4, fill=(0,0,0)):
        self.nodes = nodes
        self.fill = fill

    def __call__(self, img):
        h, w = img.size
        mask = Masks.get_ff_mask(h, w, num_v=self.nodes).astype(np.uint8)
        # masked_image = img * (1 - mask) + np.ones_like(img) * self.fill * mask
        img_array = np.array(img)
        img_array[mask==1,...] = self.fill
        # return Image.fromarray(masked_image.astype(np.uint8))
        return Image.fromarray(img_array.astype(np.uint8))


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        if isinstance(tensor, Image.Image):
            return Image.fromarray(
                (np.array(tensor) + np.random.randn(*np.array(tensor).shape)* self.std + self.mean).astype(np.uint8))
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ISONoise(object):
    def __init__(self,color_shift=0.05, intensity=0.5, random_state=None) -> None:
        self.color_shift = color_shift
        self.intensity = intensity
        self.random_state = np.random.RandomState(random_state)
    
    def __call__(self, img):
        """
        Apply poisson noise to image to simulate camera sensor noise.
        Args:
            image (numpy.ndarray): Input image, currently, only RGB, uint8 images are supported.
            color_shift (float):
            intensity (float): Multiplication factor for noise values. Values of ~0.5 are produce noticeable,
                    yet acceptable level of noise.
            random_state:
            **kwargs:
        Returns:
            numpy.ndarray: Noised image
        """
        image = np.array(img)
        if image.dtype != np.uint8:
            raise TypeError("Image must have uint8 channel type")

        one_over_255 = float(1.0 / 255.0)
        image = np.multiply(image, one_over_255, dtype=np.float32)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        _, stddev = cv2.meanStdDev(hls)

        luminance_noise = self.random_state.poisson(stddev[1] * self.intensity * 255, size=hls.shape[:2])
        color_noise = self.random_state.normal(0, self.color_shift * 360 * self.intensity, size=hls.shape[:2])

        hue = hls[..., 0]
        hue += color_noise
        hue[hue < 0] += 360
        hue[hue > 360] -= 360

        luminance = hls[..., 1]
        luminance += (luminance_noise / 255) * (1.0 - luminance)

        image = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB) * 255
        return Image.fromarray(image.astype(np.uint8))


def init_model_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


def knn_search(query, all_data, top_K=5, distance_metric="euclidean"):
    """
    Return indexes[query,top_K],dmin_distances[query,top_K]

    knn_search(clustering_model.cluster_centers_[12:15],clustering_model.cluster_centers_,top_K=10)

    knn_search(np.random.rand(3,clustering_model.cluster_centers_.shape[1])/np.sqrt(clustering_model.cluster_centers_.shape[1])
            ,clustering_model.cluster_centers_,top_K=8)
    """
    distances = cdist(query, all_data, distance_metric)
    top_indexes = np.argsort(distances, axis=1)[:, :top_K]
#     for reverse
#     top_indexes=np.argsort(distances,axis=1)[:,:top_K]
    top_distances = np.take_along_axis(distances, top_indexes, axis=1)
#     print("Indexes:")
#     print(top_indexes)
#     print("Distances:")
#     print(top_distances)
    return top_indexes, top_distances
