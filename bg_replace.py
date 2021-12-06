import argparse
import os
import os.path as osp
import cv2
import numpy as np
from paddleseg.utils import get_sys_env, logger
from deploy.infer import Predictor

def background_replace(args):
    predictor = Predictor(args)
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.img_path is not None:
        img = cv2.imread(args.img_path)
        bg = get_bg_img(args.bg_img_path, img.shape)

        comb = predictor.run(img, bg)
        save_name = osp.basename(args.img_path)
        save_path = osp.join(args.save_dir, save_name)
        cv2.imwrite(save_path, comb)
    else:
        if args.bg_video_path is not None:
            is_video_bg = True
        else:
            bg = get_bg_img(args.bg_img_path, args.input_shape)
            is_video_bg = False

        if args.video_path is not None:
            cap_video = cv2.VideoCapture(args.video_path)
            fps = cap_video.get(cv2.CAP_PROP_FPS)
            width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            save_name = osp.basename(args.video_path)
            save_name = save_name.split('.')[0]
            save_path = osp.join(args.save_dir, save_name + '.avi')

            cap_out = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,(width, height))

            if is_video_bg:
                cap_bg = cv2.VideoCapture(args.bg_video_path)
                frames_bg = cap_bg.get(cv2.CAP_PROP_FRAME_COUNT)
                current_bg = 1
            frame_num = 0
            while cap_video.isOpened():
                ret, frame = cap_video.read()
                if ret:
                    if is_video_bg:
                        ret_bg, bg = cap_bg.read()
                        if ret_bg:
                            if current_bg == frames_bg:
                                current_bg = 1
                                cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        else:
                            break
                        current_bg += 1
                    comb = predictor.run(frame, bg)
                    cap_out.write(comb)
                    frame_num += 1
                else:
                    break

            if is_video_bg:
                cap_bg.release()
            cap_video.release()
            cap_out.release()

def get_bg_img(bg_img_path, img_shape):
    if bg_img_path is None:
        bg = 255 * np.ones(img_shape)
    else:
        bg = cv2.imread(bg_img_path)
    return bg

if __name__ == "__main__":
    class input_arguments():
    	def __init__(self):
    		self.img_path = 'img.jpg'	
    		self.cfg = 'export_model/ppseg_lite_portrait_398x224_with_softmax/deploy.yaml'
    		self.bg_img_path = 'bg.jpg'
    		self.bg_video_path = None
    		self.input_shape = (192,192)
    		self.save_dir = 'output'
    		self.use_gpu = False
    		self.use_optic_flow = False
    		self.test_speed = True
    		self.soft_predict = True
    args = input_arguments()		
    background_replace(args)
