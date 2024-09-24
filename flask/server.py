# app.py
import sys
sys.path.append('.')
import os
import os.path as osp
import subprocess
import tyro
from flask import Flask, Response, render_template, request
import cv2
import numpy as np
from src.flask_pipeline import FlaskPipeline
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig

app = Flask(__name__)

ROOT = os.path.dirname(__file__)

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

# set tyro theme
tyro.extras.set_accent_color("bright_cyan")
args = tyro.cli(ArgumentConfig)

ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
if osp.exists(ffmpeg_dir):
    os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

if not fast_check_ffmpeg():
    raise ImportError(
        "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
    )
# specify configs for inference
inference_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig
crop_cfg = partial_fields(CropConfig, args.__dict__)  # use attribute of args to initial CropConfig
# global_tab_selection = None

flask_pipeline = FlaskPipeline(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg,
    args=args
)

source_image = os.path.join(ROOT, '../assets/examples/source/ins_demo.jpg')
temp_image = os.path.join(ROOT, 'frame_temp.jpg')

def process(frame):
    output_path, output_path_concat = flask_pipeline.execute_video(
        input_source_image_path=source_image,
        input_source_video_path=None,
        input_driving_video_path=None,
        input_driving_image_path=frame,
        input_driving_video_pickle_path=None,
        flag_normalize_lip=False,
        flag_relative_input=True,
        flag_do_crop_input=True,
        flag_remap_input=True,
        flag_stitching_input=True,
        animation_region="all",
        driving_option_input="expression-friendly",
        driving_multiplier=1.0,
        flag_crop_driving_video_input=False,
        # flag_video_editing_head_rotation=False,
        scale=2.3,
        vx_ratio=0.0,
        vy_ratio=-0.125,
        scale_crop_driving_video=2.2,
        vx_ratio_crop_driving_video=0.0,
        vy_ratio_crop_driving_video=-0.1,
        driving_smooth_observation_variance=3e-7,
        tab_selection="Image",
        v_tab_selection="Image"
    )
    return cv2.imread(output_path_concat, cv2.IMREAD_COLOR)

# warmup
process(source_image)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    # 从请求中获取视频流
    try:
        nparr = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(temp_image, frame)

        # 对视频帧进行处理
        processed_frame = process(temp_image)

        # 将处理后的帧转换回JPEG格式
        _, jpeg = cv2.imencode('.jpg', processed_frame)
        return Response(jpeg.tobytes(), content_type='image/jpeg')
    except Exception as e:
        print('Error processing video frame:', e)
        return Response(status=500)

if __name__ == '__main__':
    #  export CPATH=$CPATH:/usr/include/python3.8/ && python3 app_flask.py
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc', threaded=False)
