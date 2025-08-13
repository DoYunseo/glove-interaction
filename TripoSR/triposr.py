import argparse
import logging
import os
import time

import numpy as np
import rembg
import torch
import xatlas
from PIL import Image
import trimesh
import open3d as o3d

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture


def convert_mesh_to_point_cloud(mesh_path: str, save_path: str, num_points: int = 3000):
    # Load mesh with trimesh
    mesh_trimesh = trimesh.load(mesh_path, force='mesh')
    if not isinstance(mesh_trimesh, trimesh.Trimesh):
        raise ValueError(f"Loaded mesh is not a Trimesh object: {type(mesh_trimesh)}")

    # Sample points uniformly on the surface of the mesh
    points, _ = trimesh.sample.sample_surface(mesh_trimesh, count=num_points)

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Save point cloud
    o3d.io.write_point_cloud(save_path, pcd)

    return pcd


class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")


def main():
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, nargs="+", help="Path to input image(s).")
    parser.add_argument("--output-dir", default="output/", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--chunk-size", default=8192, type=int)
    parser.add_argument("--mc-resolution", default=256, type=int)
    parser.add_argument("--foreground-ratio", default=0.85, type=float)
    parser.add_argument("--pretrained-model-name-or-path", default="stabilityai/TripoSR")
    parser.add_argument("--model-save-format", default="obj", choices=["obj", "glb"])
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"

    timer = Timer()
    timer.start("Initializing model")
    model = TSR.from_pretrained(
        args.pretrained_model_name_or_path,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(args.chunk_size)
    model.to(device)
    timer.end("Initializing model")

    timer.start("Processing images")
    rembg_session = rembg.new_session()
    images = []
    for image_path in args.image:
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(output_dir, image_id)
        os.makedirs(save_path, exist_ok=True)

        image = remove_background(Image.open(image_path), rembg_session)
        image = resize_foreground(image, args.foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        image.save(os.path.join(save_path, f"input.png"))
        images.append((image, image_id))
        logging.info(f"[✓] Saved results to: {save_path}")
    timer.end("Processing images")

    for i, (image, image_id) in enumerate(images):
        save_path = os.path.join(output_dir, image_id)
        logging.info(f"Running image {i + 1}/{len(images)} ...")

        timer.start("Running model")
        with torch.no_grad():
            scene_codes = model([image], device=device)
        timer.end("Running model")

        timer.start("Extracting mesh")
        meshes = model.extract_mesh(scene_codes, True, resolution=args.mc_resolution)
        timer.end("Extracting mesh")

        mesh_path = os.path.join(save_path, f"mesh.{args.model_save_format}")
        meshes[0].export(mesh_path)
        logging.info(f"Saved mesh to {mesh_path}")

        timer.start("Converting mesh to point cloud")
        pcd_path = os.path.join(save_path, "point_cloud.ply")
        pcd = convert_mesh_to_point_cloud(mesh_path, pcd_path)
        timer.end("Converting mesh to point cloud")

        # Visualize point cloud
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
