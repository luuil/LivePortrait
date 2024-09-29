# coding: utf-8

"""
Pipeline for gradio
"""

import os.path as osp
import os
import cv2
from rich.progress import track
import numpy as np
import torch
from PIL import Image

from .config.argument_config import ArgumentConfig
from .live_portrait_pipeline import LivePortraitPipeline
from .utils.io import load_img_online, load_video, resize_to_limit
from .utils.filter import smooth
from .utils.rprint import rlog as log
from .utils.crop import prepare_paste_back, paste_back
from .utils.camera import get_rotation_matrix
from .utils.video import get_fps, has_audio_stream, concat_frames, images2video, add_audio_to_video
from .utils.helper import is_square_video, mkdir, dct2device, basename, calc_motion_multiplier
from .utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio


def update_args(args, user_args):
    """update the args according to user inputs
    """
    for k, v in user_args.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args


class TalkingheadPipeline(LivePortraitPipeline):
    """Liveportrait for talkinghead
    """

    def __init__(self, inference_cfg, crop_cfg, args: ArgumentConfig):
        super().__init__(inference_cfg, crop_cfg)
        self.args = args
        self.source = None
        self.source_info = None
        self.is_first_frame = True

    @torch.no_grad()
    def execute_video(
        self,
        input_source_image_path=None,
        input_source_video_path=None,
        input_driving_video_path=None,
        input_driving_image_path=None,
        input_driving_video_pickle_path=None,
        flag_normalize_lip=False,
        flag_relative_input=True,
        flag_do_crop_input=True,
        flag_remap_input=True,
        flag_stitching_input=True,
        animation_region="all",
        driving_option_input="pose-friendly",
        driving_multiplier=1.0,
        flag_crop_driving_video_input=True,
        # flag_video_editing_head_rotation=False,
        scale=2.3,
        vx_ratio=0.0,
        vy_ratio=-0.125,
        scale_crop_driving_video=2.2,
        vx_ratio_crop_driving_video=0.0,
        vy_ratio_crop_driving_video=-0.1,
        driving_smooth_observation_variance=3e-7,
        tab_selection=None,
        v_tab_selection=None
    ):
        """ for video-driven portrait animation or video editing
        """
        if tab_selection == 'Image':
            input_source_path = input_source_image_path
        elif tab_selection == 'Video':
            input_source_path = input_source_video_path
        else:
            input_source_path = input_source_image_path

        if v_tab_selection == 'Video':
            input_driving_path = input_driving_video_path
        elif v_tab_selection == 'Image':
            input_driving_path = input_driving_image_path
        elif v_tab_selection == 'Pickle':
            input_driving_path = input_driving_video_pickle_path
        else:
            input_driving_path = input_driving_video_path

        if input_source_path is not None and input_driving_path is not None:
            if osp.exists(input_driving_path) and v_tab_selection == 'Video' and not flag_crop_driving_video_input and is_square_video(input_driving_path) is False:
                flag_crop_driving_video_input = True
                log("The driving video is not square, it will be cropped to square automatically.")

            args_user = {
                'source': input_source_path,
                'driving': input_driving_path,
                'flag_normalize_lip' : flag_normalize_lip,
                'flag_relative_motion': flag_relative_input,
                'flag_do_crop': flag_do_crop_input,
                'flag_pasteback': flag_remap_input,
                'flag_stitching': flag_stitching_input,
                'animation_region': animation_region,
                'driving_option': driving_option_input,
                'driving_multiplier': driving_multiplier,
                'flag_crop_driving_video': flag_crop_driving_video_input,
                'scale': scale,
                'vx_ratio': vx_ratio,
                'vy_ratio': vy_ratio,
                'scale_crop_driving_video': scale_crop_driving_video,
                'vx_ratio_crop_driving_video': vx_ratio_crop_driving_video,
                'vy_ratio_crop_driving_video': vy_ratio_crop_driving_video,
                'driving_smooth_observation_variance': driving_smooth_observation_variance,
            }
            # update config from user input
            self.args = update_args(self.args, args_user)
            self.live_portrait_wrapper.update_config(self.args.__dict__)
            self.cropper.update_config(self.args.__dict__)

            output_path, output_path_concat = self.execute(self.args)

            log("Run successfully!")
            return output_path, output_path_concat
        else:
            raise Exception("Please upload the source portrait or source video, and driving video 🤗🤗🤗")

    def excute_frame(
            self,
            input_source_image,
            input_driving_image,
            flag_normalize_lip=False,
            flag_relative_input=True,
            flag_do_crop_input=True,
            flag_remap_input=True,
            flag_stitching_input=True,
            animation_region="all",
            driving_option_input="expression-friendly",
            driving_multiplier=1.0,
            flag_crop_driving_video_input=False,
            scale=2.3,
            vx_ratio=0.0,
            vy_ratio=-0.125,
            scale_crop_driving_video=2.2,
            vx_ratio_crop_driving_video=0.0,
            vy_ratio_crop_driving_video=-0.1,
            driving_smooth_observation_variance=3e-7
        ):
        args_user = {
                'source': input_source_image,
                'driving': input_driving_image,
                'flag_normalize_lip' : flag_normalize_lip,
                'flag_relative_motion': flag_relative_input,
                'flag_do_crop': flag_do_crop_input,
                'flag_pasteback': flag_remap_input,
                'flag_stitching': flag_stitching_input,
                'animation_region': animation_region,
                'driving_option': driving_option_input,
                'driving_multiplier': driving_multiplier,
                'flag_crop_driving_video': flag_crop_driving_video_input,
                'scale': scale,
                'vx_ratio': vx_ratio,
                'vy_ratio': vy_ratio,
                'scale_crop_driving_video': scale_crop_driving_video,
                'vx_ratio_crop_driving_video': vx_ratio_crop_driving_video,
                'vy_ratio_crop_driving_video': vy_ratio_crop_driving_video,
                'driving_smooth_observation_variance': driving_smooth_observation_variance,
            }
        # update config from user input
        self.args = update_args(self.args, args_user)
        self.live_portrait_wrapper.update_config(self.args.__dict__)
        self.cropper.update_config(self.args.__dict__)

        output_path, output_path_concat = self._execute_frame(self.args)
        return output_path, output_path_concat

    def _execute_frame(self, args: ArgumentConfig):
        # for convenience
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        device = self.live_portrait_wrapper.device
        crop_cfg = self.cropper.crop_cfg

        ######## load source input ########
        flag_is_source_video = False
        source_fps = None
        if isinstance(args.source, Image.Image):
            flag_is_source_video = False
            img_rgb = np.array(args.source)
            img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)
            source_rgb_lst = [img_rgb]
        else:  # source input is an unknown format
            raise Exception(f"Unknown source format: {args.source}")

        ######## process driving info ########
        flag_load_from_template = False
        driving_rgb_crop_256x256_lst = None
        flag_is_driving_video = False

        driving_img_rgb = np.array(args.driving)
        output_fps = 25
        driving_rgb_lst = [driving_img_rgb]

        ######## make motion template ########
        driving_n_frames = len(driving_rgb_lst)
        if flag_is_source_video and flag_is_driving_video:
            n_frames = min(len(source_rgb_lst), driving_n_frames)  # minimum number as the number of the animated frames
            driving_rgb_lst = driving_rgb_lst[:n_frames]
        elif flag_is_source_video and not flag_is_driving_video:
            n_frames = len(source_rgb_lst)
        else:
            n_frames = driving_n_frames
        if inf_cfg.flag_crop_driving_video:
            log('flag_crop_driving_video')
            ret_d = self.cropper.crop_driving_video(driving_rgb_lst)
            log(f'Driving video/image is cropped, {len(ret_d["frame_crop_lst"])} frames are processed.')
            if len(ret_d["frame_crop_lst"]) != n_frames and flag_is_driving_video:
                n_frames = min(n_frames, len(ret_d["frame_crop_lst"]))
            driving_rgb_crop_lst, driving_lmk_crop_lst = ret_d['frame_crop_lst'], ret_d['lmk_crop_lst']
            driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_crop_lst]
        else:
            log('force to resize to 256x256')
            driving_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(driving_rgb_lst)
            driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]  # force to resize to 256x256
        #######################################
        # Image.fromarray(driving_rgb_crop_256x256_lst[0], 'RGB').save('driving_rgb_crop_256x256.png')
        c_d_eyes_lst, c_d_lip_lst = self.live_portrait_wrapper.calc_ratio(driving_lmk_crop_lst)

        # save the motion template
        I_d_lst = self.live_portrait_wrapper.prepare_videos(driving_rgb_crop_256x256_lst)
        driving_template_dct = self.make_motion_template(I_d_lst, c_d_eyes_lst, c_d_lip_lst, output_fps=output_fps)

        c_d_eyes_lst = c_d_eyes_lst*n_frames
        c_d_lip_lst = c_d_lip_lst*n_frames

        ######## prepare for pasteback ########
        I_p_pstbk_lst = None
        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            I_p_pstbk_lst = []

        I_p_lst = []
        R_d_0, x_d_0_info = None, None
        flag_normalize_lip = inf_cfg.flag_normalize_lip  # not overwrite
        flag_source_video_eye_retargeting = inf_cfg.flag_source_video_eye_retargeting  # not overwrite
        lip_delta_before_animation, eye_delta_before_animation, mask_ori_float = None, None, None

        ######## process source info ########
        if self.is_first_frame:
            if inf_cfg.flag_do_crop:
                crop_info = self.cropper.crop_source_image(source_rgb_lst[0], crop_cfg)
                if crop_info is None:
                    raise Exception("No face detected in the source image!")
                source_lmk = crop_info['lmk_crop']
                img_crop_256x256 = crop_info['img_crop_256x256']
            else:
                source_lmk = self.cropper.calc_lmk_from_cropped_image(source_rgb_lst[0])
                img_crop_256x256 = cv2.resize(source_rgb_lst[0], (256, 256))  # force to resize to 256x256
            I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
            x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
            x_c_s = x_s_info['kp']
            R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
            x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

            # let lip-open scalar to be 0 at first
            if flag_normalize_lip and inf_cfg.flag_relative_motion and source_lmk is not None:
                c_d_lip_before_animation = [0.]
                combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
                if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_normalize_threshold:
                    lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(source_rgb_lst[0].shape[1], source_rgb_lst[0].shape[0]))

            self.crop_info = crop_info
            self.img_crop_256x256 = img_crop_256x256
            self.x_s_info = x_s_info
            self.x_c_s = x_c_s
            self.R_s = R_s
            self.f_s = f_s
            self.x_s = x_s
            self.x_s = x_s
            self.mask_ori_float = mask_ori_float
            self.lip_delta_before_animation = lip_delta_before_animation

            self.is_first_frame = False
        else:
            crop_info = self.crop_info
            img_crop_256x256 = self.img_crop_256x256
            x_s_info = self.x_s_info
            x_c_s = self.x_c_s
            R_s = self.R_s
            f_s = self.f_s
            x_s = self.x_s
            mask_ori_float = self.mask_ori_float
            lip_delta_before_animation = self.lip_delta_before_animation


        ######## animate ########
        if flag_is_driving_video or (flag_is_source_video and not flag_is_driving_video):
            log(f"The animated video consists of {n_frames} frames.")
        else:
            log(f"The output of image-driven portrait animation is an image.")
        for i in track(range(n_frames), description='🚀Animating...', total=n_frames, disable=True):
            if flag_is_source_video and not flag_is_driving_video:
                x_d_i_info = driving_template_dct['motion'][0]
            else:
                x_d_i_info = driving_template_dct['motion'][i]
            x_d_i_info = dct2device(x_d_i_info, device)
            R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys

            if i == 0:  # cache the first frame
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info.copy()

            delta_new = x_s_info['exp'].clone()
            if inf_cfg.flag_relative_motion:
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
                else:
                    R_new = R_s
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
                    delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device))
                elif inf_cfg.animation_region == "lip":
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device)))[:, lip_idx, :]
                elif inf_cfg.animation_region == "eyes":
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - 0))[:, eyes_idx, :]
                if inf_cfg.animation_region == "all":
                    scale_new = x_s_info['scale'] if flag_is_source_video else x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                else:
                    scale_new = x_s_info['scale']
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    t_new = x_s_info['t'] if flag_is_source_video else x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
                else:
                    t_new = x_s_info['t']
            else:
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    R_new = R_d_i
                else:
                    R_new = R_s
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
                    for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                        delta_new[:, idx, :] = x_d_i_info['exp'][:, idx, :]
                    delta_new[:, 3:5, 1] = x_d_i_info['exp'][:, 3:5, 1]
                    delta_new[:, 5, 2] = x_d_i_info['exp'][:, 5, 2]
                    delta_new[:, 8, 2] = x_d_i_info['exp'][:, 8, 2]
                    delta_new[:, 9, 1:] = x_d_i_info['exp'][:, 9, 1:]
                elif inf_cfg.animation_region == "lip":
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = x_d_i_info['exp'][:, lip_idx, :]
                elif inf_cfg.animation_region == "eyes":
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = x_d_i_info['exp'][:, eyes_idx, :]
                scale_new = x_s_info['scale']
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    t_new = x_d_i_info['t']
                else:
                    t_new = x_s_info['t']

            t_new[..., 2].fill_(0)  # zero tz
            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            if inf_cfg.flag_relative_motion and inf_cfg.driving_option == "expression-friendly" and not flag_is_source_video and flag_is_driving_video:
                if i == 0:
                    x_d_0_new = x_d_i_new
                    motion_multiplier = calc_motion_multiplier(x_s, x_d_0_new)
                    # motion_multiplier *= inf_cfg.driving_multiplier
                x_d_diff = (x_d_i_new - x_d_0_new) * motion_multiplier
                x_d_i_new = x_d_diff + x_s

            # Algorithm 1:
            if not inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
                # without stitching or retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new += lip_delta_before_animation
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
                else:
                    pass
            elif inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
                # with stitching and without retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new) + lip_delta_before_animation
                else:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
            else:
                eyes_delta, lip_delta = None, None
                if inf_cfg.flag_eye_retargeting and source_lmk is not None:
                    c_d_eyes_i = c_d_eyes_lst[i]
                    combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eyes_i, source_lmk)
                    # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                    eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor)
                if inf_cfg.flag_lip_retargeting and source_lmk is not None:
                    c_d_lip_i = c_d_lip_lst[i]
                    combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
                    # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                    lip_delta = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor)

                if inf_cfg.flag_relative_motion:  # use x_s
                    x_d_i_new = x_s + \
                        (eyes_delta if eyes_delta is not None else 0) + \
                        (lip_delta if lip_delta is not None else 0)
                else:  # use x_d,i
                    x_d_i_new = x_d_i_new + \
                        (eyes_delta if eyes_delta is not None else 0) + \
                        (lip_delta if lip_delta is not None else 0)

                if inf_cfg.flag_stitching:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)

            x_d_i_new = x_s + (x_d_i_new - x_s) * inf_cfg.driving_multiplier
            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                # TODO: the paste back procedure is slow, considering optimize it using multi-threading or GPU
                I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], source_rgb_lst[0], mask_ori_float)
                I_p_pstbk_lst.append(I_p_pstbk)

        ######### build the final concatenation result #########
        # driving frame | source frame | generation
        frames_concatenated = concat_frames(driving_rgb_crop_256x256_lst, [img_crop_256x256], I_p_lst)
        wfp_concat = frames_concatenated[0]
        if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
            wfp = I_p_pstbk_lst[0]
        else:
            wfp = frames_concatenated[0]
        return wfp, wfp_concat
