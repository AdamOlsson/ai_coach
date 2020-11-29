import numpy as np
import cv2

class FactorCrop(object):

    def __init__(self, factor, dest_size=368):
        self.factor = factor
        self.dest_size = dest_size

    def __call__(self, sample):

        def FactorCropVideo():
            video = sample['data']

            del sample['data']

            min_dimension = np.min(video.shape[1:3])
            scale_factor = float(self.dest_size) / min_dimension

            video_resized_buffer = np.empty((video.shape[0], int(video.shape[1]*scale_factor), int(video.shape[2]*scale_factor), video.shape[3]))

            for idx, frame in enumerate(video[:]):
                video_resized_buffer[idx] = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
            
            del video

            t, h, w, c = video_resized_buffer.shape

            # TODO: find out the purpose of this
            h_new = int(np.ceil( h / self.factor))*self.factor
            w_new = int(np.ceil( w / self.factor))*self.factor

            # Due to memory constarints, we crop frames one by one
            video_cropped = np.zeros([t, h_new, w_new, c], dtype=video_resized_buffer.dtype)
            video_cropped[:, :h, :w, :] = video_resized_buffer

            sample['data'] = video_cropped

        def FactorCropImage():
            image = sample['data']

            min_dimension = np.min(image.shape[:2])
            scale_factor = float(self.dest_size) / min_dimension

            image_resized = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

            h, w, c = image_resized.shape
            
            # TODO: find out the purpose of this
            h_new = int(np.ceil( h / self.factor))*self.factor
            w_new = int(np.ceil( w / self.factor))*self.factor

            image_cropped = np.zeros([h_new, w_new, c], dtype=image_resized.dtype)
            image_cropped[:h, :w, :] = image_resized

            sample['data'] = image_cropped


        type = sample['type']
        
        if type == 'image':     # Assuming 3 dims [H, W, C]
            FactorCropImage()
            return sample
        elif type == 'video':   # Assuming 4 dims [T, H, W, C]
            FactorCropVideo()
            return sample
        else:                   # Not yet implemented
            raise NotImplementedError("Data of type '{}' has yet not been implemented with this transformer.".format(type))
        