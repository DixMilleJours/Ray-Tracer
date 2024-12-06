from utils import *
from ray import *
from cli import render
from PIL import Image
from obj_reader import *


grass = Material( k_d = 0.3* vec([0.44, 0.74, 0.12]),k_a=5* vec([0.44, 0.74, 0.12]),k_s=0., p=10, k_m=0.01)
bull = Material(vec([0.7, 0.45, 0.1]), k_s=0., p=90, k_m=0.)
# blue = Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
blue = Material(vec([0, 0, 0]), k_a= 0.1, k_m=0.3, k_t = 0.95, n = 1.1)
blue = Material(vec([0, 0, 0]), k_a= 0.1, k_m=0.3, k_t = 0.1, n = 1.1)
gray = Material(vec([0.2, 0.2, 0.2]))
glass_material = Material(
    k_d = vec([0, 0, 0]),    # Very slight diffuse color (almost clear)
    k_s = 0.2,                 # Slight specular highlight
    k_m = 0.9,                 # Slight mirror reflection
    k_t = 0.95,  # High transparency
    n = 1.5,                  # Typical index of refraction for glass (ranges 1.45-1.9)
)
sun = Material(
    k_d=vec([0,0,0]),  # Pure red diffuse color
    k_s= 0.,              # Specular coefficient (moderate shininess)
    p=10.,                # Specular exponent (sharp highlights)
    k_m=0.3,              # Mirror reflection coefficient (slightly reflective)
    k_a= 7 * vec([1,0.3,0]),  # Ambient coefficient (dark red ambient color)
    k_t=0.0,  # Transparency coefficient (opaque)
    n=1.0,                # Index of refraction (default for non-refractive material)
)

# Read the triangle meshes for two cubes from the updated cube.obj
vs_list = []

# Since the OBJ contains two meshes, we get the list of vertices for each mesh separately
meshes = read_obj_extended(open("cow.obj"))

for mesh in meshes:
    # Scale down to fit the scene (optional, depending on scene size)
    vs_list.extend(mesh['vertices'] * 0.5)  # Scale all vertices

scene = AcceleratedScene([
    Sphere(vec([0,-0.2,1.5]), 0.3, glass_material),
    Sphere(vec([-1.7,0.6,0.2]), 0.3, sun),
    Sphere(vec([0,-40,0]), 39.5, grass),
] + [
    Triangle(vs, bull) for vs in vs_list
] + [
    Disc(vec([0.3,0.2,0.6]), 0.3, vec([0.,0.,1.]), Material(vec([0, 0, 0]), k_m = 0.5, k_a = 0.1, k_t = 0.95, f = 0.3)),
    Disc(vec([-0.6,0.2,0.8]), 0.3, vec([0.,0.,1.]), Material(vec([0, 0, 0]), k_m = 0.5, k_a = 0.1, k_t = 0.95, f = -0.3)),
]

)

lights = [
    PointLight(vec([-7.7,0.3,0.2]), 0.3*vec([300,300,300])),
    AmbientLight(0.1),
]

camera = Camera(vec([0.8,1.2,6]), target=vec([0,-0.4,0]), vfov=24, aspect=16/9)

start_time = time.time()  # Record the start time

# img = render_image_with_progress(camera, scene, lights, 160, 90)
# img = render_image_with_progress(camera, scene, lights, 320, 180)
img = render_image_with_progress(camera, scene, lights, 1920, 1080)

cam_img_ui8 = to_srgb8(img / 1.0)
Image.fromarray(cam_img_ui8, 'RGB').save('3spheres.png')

end_time = time.time()  # Record the end time

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.6f} seconds")