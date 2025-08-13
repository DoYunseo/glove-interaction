import trimesh
import open3d as o3d

# 1. .obj 파일 불러오기
mesh = trimesh.load('output/mouse/mesh.obj')

# 2. Trimesh → Open3D TriangleMesh로 변환
mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)

# 3. 표면에서 point cloud 샘플링 (예: 5000개)
pcd = mesh_o3d.sample_points_uniformly(number_of_points=10000)

# 4. 시각화 (선택)
o3d.visualization.draw_geometries([pcd])

# 5. 저장 (선택)
o3d.io.write_point_cloud("output/mouse/pointcloud.ply", pcd)
