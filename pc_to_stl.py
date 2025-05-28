import sys
import numpy as np
import open3d as o3d
import os
import time

# Define parameters for plaque and cube creation
plaque_offset = 0.05          # Distance behind the front of the face to place the plaque
plaque_thickness = 0.005      # Thickness of the plaque in meters
scale_factor = 2.25  # e.g., # Scale up the mesh so it isn't mucking tiny

watch_folder = "/Users/donovanlecours/PycharmProjects/mesh-generator"  # <- Change this to your actual watched folder path
trigger_file = "scan_done.txt"
trigger_path = os.path.join(watch_folder, trigger_file)
pcd_path = os.path.join(watch_folder, "face.ply")
done_path = os.path.join(watch_folder, "mesh_done.txt")

print(f"Waiting for '{trigger_file}' in '{watch_folder}'...")
while not os.path.exists(trigger_path):
    time.sleep(1)  # Wait 1 second between checks

print(f"Detected '{trigger_file}', starting processing...")
os.remove(trigger_path)

# Read the point cloud from file
try:
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        print("Point cloud is empty. Exiting.")
        sys.exit()
except Exception as e:
    print(f"Error reading point cloud: {e}")
    sys.exit()

# Get the bounding box for the point cloud to define plaque size
bbox = pcd.get_axis_aligned_bounding_box()
min_bound = bbox.min_bound
max_bound = bbox.max_bound

# Define Z cutoff for cropping (plaque position)
plaque_z = min_bound[2] + plaque_offset

# Convert point cloud to numpy array and filter
all_points = np.asarray(pcd.points)
mask = all_points[:, 2] <= plaque_z
filtered_points = all_points[mask]

# Create a new filtered point cloud
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

# Estimate normals
filtered_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=30))
filtered_pcd.orient_normals_consistent_tangent_plane(k=50)

# Reconstruct mesh using Poisson
rec_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(filtered_pcd, depth=10)
rec_mesh = rec_mesh.simplify_quadric_decimation(5000)

# Clean mesh
rec_mesh.compute_vertex_normals()
rec_mesh.remove_duplicated_vertices()
rec_mesh.remove_duplicated_triangles()
rec_mesh.remove_non_manifold_edges()

# Get updated bounds from cropped mesh
cropped_bbox = rec_mesh.get_axis_aligned_bounding_box()
crop_min = cropped_bbox.min_bound
crop_max = cropped_bbox.max_bound

# Create a plaque cube with dimensions of the cropped mesh
cube_width = crop_max[0] - crop_min[0]
cube_depth = crop_max[1] - crop_min[1]
cube_height = plaque_thickness

cube = o3d.geometry.TriangleMesh.create_box(cube_width, cube_depth, cube_height)

# Position cube right behind the cropped mesh
cube.translate(np.array([crop_min[0], crop_min[1], crop_min[2] - plaque_thickness]))

# Optional: Color the plaque
cube.paint_uniform_color([0.3, 0.3, 0.3])

# Combine mesh and plaque
combined_mesh = rec_mesh


# Center mesh to world origin
center = combined_mesh.get_center()
combined_mesh.translate(-center)

# Final cleanup
combined_mesh.compute_vertex_normals()
combined_mesh.remove_duplicated_vertices()
combined_mesh.remove_duplicated_triangles()
combined_mesh.remove_non_manifold_edges()

###MESH SCALING
combined_mesh.scale(scale_factor, center=combined_mesh.get_center())


# Export
output_path = os.path.join(watch_folder, "output_with_plaque.obj")
o3d.io.write_triangle_mesh(output_path, combined_mesh)
print(f"Exported watertight face mesh with plaque to OBJ: '{output_path}'")

# Optional visualization
#o3d.visualization.draw_geometries([filtered_pcd, combined_mesh])

# Signal completion by writing a trigger file
with open(done_path, "w") as f:
    f.write("Mesh processing complete.\n")
print(f"Wrote completion signal to '{done_path}'")
