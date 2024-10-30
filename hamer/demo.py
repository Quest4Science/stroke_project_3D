from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from OpenGL import GL
from OpenGL.GL import *
import os
from time import perf_counter

os.environ["PYOPENGL_PLATFORM"] = "egl"

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils.geometry import aa_to_rotmat, perspective_projection
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images',
                       help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo',
                       help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False,
                       help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True,
                       help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False,
                       help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0,
                       help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet',
                       choices=['vitdet', 'regnety'],
                       help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--v_path', type=str, required=True,
                       help='Video Path')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    else:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Keypoint detector
    cpm = ViTPoseModel(device)

    # Setup renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Create output directory
    os.makedirs(args.out_folder, exist_ok=True)
    joints_folder = os.path.join(args.out_folder, 'keypoints')
    os.makedirs(joints_folder, exist_ok=True)
    # Open video
    cap = cv2.VideoCapture(args.v_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.v_path}")

    frame_id = 0
    times = []

    while cap.isOpened():
        success, img_cv2 = cap.read()
        if not success:
            break

        frame_id += 1
        start = perf_counter()

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Left hand
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(0)

            # Right hand
            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                               scaled_focal_length).detach().cpu().numpy()
            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:, None, None] / 255) / (
                            DEFAULT_STD[:, None, None] / 255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:, None, None] / 255) + (
                            DEFAULT_MEAN[:, None, None] / 255)
                input_patch = input_patch.permute(1, 2, 0).numpy()

                # Get vertices and cam_t
                vertices = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                vertices[:, 0] = (2 * is_right - 1) * vertices[:, 0]
                cam_t = pred_cam_t_full[n]

                regression_img = renderer(vertices,
                                          cam_t,
                                          batch['img'][n],
                                          mesh_base_color=LIGHT_BLUE,
                                          scene_bg_color=(1, 1, 1))

                if args.side_view:
                    side_img = renderer(vertices,
                                        cam_t,
                                        white_img,
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        side_view=True)
                    final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                else:
                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                cv2.imwrite(os.path.join(args.out_folder, f'{frame_id}_{person_id}.png'), 255 * final_img[:, :, ::-1])

                # 在保存NPZ文件的部分进行修改
                keypoints_3d = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]

                # 计算世界坐标系下的关键点位置
                # 将相机坐标系的关键点加上相机平移向量得到世界坐标系下的位置
                keypoints_3d_world = keypoints_3d + cam_t[None, :]  # [None, :] 用于广播

                # 修改 output 字典，增加世界坐标系下的关键点
                output = {
                    'vertices_3d': vertices,
                    'keypoints_3d': keypoints_3d,  # 相机坐标系
                    'keypoints_3d_world': keypoints_3d_world,  # 世界坐标系
                    'camera_translation': cam_t,
                    'camera_parameters': {
                        'focal_length': scaled_focal_length,
                        'center': [float(img_size[n][0] / 2.), float(img_size[n][1] / 2.)]
                    },
                    'is_right': is_right
                }

                # Add hand pose if available
                if 'hand_pose' in out:
                    output['hand_pose'] = out['hand_pose'][n].detach().cpu().numpy()
                if 'root_pose' in out:
                    output['root_pose'] = out['root_pose'][n].detach().cpu().numpy()

                np.savez(os.path.join(joints_folder, f'{frame_id}_{person_id}.npz'), **output)

                # Add to lists for full frame rendering
                all_verts.append(vertices)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

                # Save mesh if requested
                if args.save_mesh:
                    tmesh = renderer.vertices_to_trimesh(vertices, cam_t, LIGHT_BLUE, is_right=is_right)
                    tmesh.export(os.path.join(args.out_folder, f'{frame_id}_{person_id}.obj'))

            # Render full frame
            if args.full_frame and len(all_verts) > 0:
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                cam_view = renderer.render_rgba_multiple(
                    all_verts,
                    cam_t=all_cam_t,
                    render_res=img_size[n],
                    is_right=all_right,
                    **misc_args
                )

                input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
                input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :,
                                                                                                          3:]

                cv2.imwrite(os.path.join(args.out_folder, f'{frame_id}_all.jpg'), 255 * input_img_overlay[:, :, ::-1])

            # Calculate and display FPS
        times.append(perf_counter() - start)
        if frame_id % 10 == 0:
            avg_fps = 1.0 / (sum(times) / len(times))
            print(f'Frame {frame_id}, Average FPS: {avg_fps:.2f}')

    cap.release()
    cv2.destroyAllWindows()

    if times:
        print(f'Final average FPS = {1.0 / (sum(times) / len(times)):.2f}')


if __name__ == '__main__':
    main()