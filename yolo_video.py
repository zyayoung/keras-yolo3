# import os 
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import argparse
from PIL import Image

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use'
    )

    parser.add_argument(
        '--track', default=False, action="store_true",
        help='Tracking mode, will not perform detection'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if "input" in FLAGS:
        if FLAGS.track:
            """
            Tracking mode
            """
            from tracker import track_video
            print("Tracking mode")
            track_video(FLAGS.input, FLAGS.output)
        else:
            from yolo import YOLO, detect_video
            detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
