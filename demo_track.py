import os
import argparse
import logging
import textwrap

import torch
import torch.backends.cudnn as cudnn
import cv2

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
# from detectron2.engine import launch
from detectron2.utils import comm

from datasets.data import ValTransform
from tracker.eval.timer import Timer
from tracker.byte_tracker import BYTETracker
from tracker.sparse_tracker import SparseTracker
from utils.model_utils import fuse_model
from utils.visualize import plot_tracking

from track import default_track_setup


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--video-input", type=str, required=True, help="path to video file")
    parser.add_argument(
        "opts",
        help=textwrap.dedent("""
            Modify config options at the end of the command. For Yacs configs, use
            space-separated "PATH.KEY VALUE" pairs.
            For python-based LazyConfig, use "path.key=value".
        """),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def do_track(cfg, model, video_input):
    logger = logging.getLogger("detectron2")
    if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
        logger.info("Run evaluation with EMA.")
    else:
        logger.info("Run evaluation without EMA.")

    cudnn.benchmark = False

    # set environment variables for distributed inference
    file_name = os.path.join(cfg.train.output_dir, cfg.track.experiment_name)
    if comm.is_main_process():
        os.makedirs(file_name, exist_ok=True)
    results_folder = os.path.join(file_name, "track_results")
    os.makedirs(results_folder, exist_ok=True)    

    # build evaluator (skip)

    model.eval()
    if cfg.track.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    # start track
    half = cfg.track.fp16
    tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
    # model = model.eval()
    if half:
        model = model.half()

    if cfg.track.byte:
        tracker = BYTETracker(cfg.track)
    elif cfg.track.deep:
        tracker = SparseTracker(cfg.track)

    ori_thresh = cfg.track.track_thresh
    ori_track_buffer = cfg.track.track_buffer
    video_id = 0

    cap = cv2.VideoCapture(video_input)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    save_path = "result.mp4"
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    # mapper = MOTtestMapper(
    #     test_size = cfg.dataloader.test.test_size,
    #     preproc = ValTransform(
    #         rgb_means=(0.485, 0.456, 0.406),
    #         std=(0.229, 0.224, 0.225),
    #     ),
    # )
    img_size = cfg.dataloader.test.test_size
    transform = ValTransform(
        rgb_means=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    timer = Timer()
    frame_id = 0
    results = []
    while True:
        thpt = 1. / max(1e-5, timer.average_time)
        if frame_id % 20 == 0:
            logger.info(f"Processing frame {frame_id} ({thpt:.2f} fps)")
        ret_val, frame = cap.read()
        if ret_val:

            with torch.no_grad():
                frame_tr, _ = transform(frame, None, img_size)
                frame_tr = torch.from_numpy(frame_tr).type(tensor_type)
                # frame_tr = torch.tensor(frame_tr, device=model.device)
                # if half:
                #     frame_tr = frame_tr.half()
                height, width = frame.shape[:2]
                img_data = [{
                    "height": height,
                    "width": width,
                    "image": frame_tr,
                    "ori_img": frame,
                }]

                # run model
                timer.tic()
                outputs = model(img_data)

            if outputs[0]["instances"] is not None:
                online_targets = tracker.update(
                    outputs[0]["instances"], img_data[0]["ori_img"]
                )
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > cfg.track.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))
                timer.toc()
                online_im = plot_tracking(
                    img_data[0]["ori_img"], online_tlwhs, online_ids, frame_id=frame_id, fps=thpt
                )
            else:
                timer.toc()
                online_im = img_data[0]["ori_img"]
            
            vid_writer.write(online_im)

        else:
            break

        frame_id += 1


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_track_setup(cfg, args)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model.device = torch.device(cfg.train.device)
    # model = create_ddp_model(model)

    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

    do_track(cfg, model, args.video_input)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
    # launch(
    #     main,
    #     1,  # args.num_gpus
    #     num_machines=1,  # args.num_machines
    #     machine_rank=0,  # args.machine_rank
    #     dist_url=None,  # args.dist_url
    #     args=(args,),
    # )