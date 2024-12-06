import numpy as np

from utils import *


import time

from datetime import timedelta

"""
Core implementation of the ray tracer.  This module contains the classes (Sphere, Mesh, etc.)
that define the contents of scenes, as well as classes (Ray, Hit) and functions (shade) used in 
the rendering algorithm, and the main entry point `render_image`.

In the documentation of these classes, we indicate the expected types of arguments with a
colon, and use the convention that just writing a tuple means that the expected type is a
NumPy array of that shape.  Implementations can assume these types are preconditions that
are met, and if they fail for other type inputs it's an error of the caller.  (This might 
not be the best way to handle such validation in industrial-strength code but we are adopting
this rule to keep things simple and efficient.)
"""


class Ray:

    def __init__(self, origin, direction, start=0., end=np.inf):
        """Create a ray with the given origin and direction.

        Parameters:
          origin : (3,) -- the start point of the ray, a 3D point
          direction : (3,) -- the direction of the ray, a 3D vector (not necessarily normalized)
          start, end : float -- the minimum and maximum t values for intersections
        """
        # Convert these vectors to double to help ensure intersection
        # computations will be done in double precision
        self.origin = np.array(origin, np.float64)
        self.direction = np.array(direction, np.float64)
        self.start = start
        self.end = end


class Material:

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None, k_t = 0. ,n=10., R0 = None, f = None):
        """Create a new material with the given parameters.

        Parameters:
          k_d : (3,) -- the diffuse coefficient
          k_s : (3,) or float -- the specular coefficient
          p : float -- the specular exponent
          k_m : (3,) or float -- the mirror reflection coefficient
          k_a : (3,) -- the ambient coefficient (defaults to match diffuse color)
          k_t : (3,) -- the transparency coefficient
          n : float -- the index of refraction
          f : focus
          opticalcenter : (3,) -- the optical center of the lens
        """
        self.k_d = k_d
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else k_d
        self.k_t = k_t
        self.n = n
        self.R0 = R0 if R0 is not None else ((n - 1) / (n + 1)) ** 2
        self.f = f

class Hit:

    def __init__(self, t, point=None, normal=None, material=None, opticalcenter = None):
        """Create a Hit with the given data.

        Parameters:
          t : float -- the t value of the intersection along the ray
          point : (3,) -- the 3D point where the intersection happens
          normal : (3,) -- the 3D outward-facing unit normal to the surface at the hit point
          material : (Material) -- the material of the surface
          opticalcenter : (3,) -- the optical center of the lens
        """
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material
        self.opticalcenter = opticalcenter

# Value to represent absence of an intersection
no_hit = Hit(np.inf)


class Sphere:

    def __init__(self, center, radius, material):
        """Create a sphere with the given center and radius.

        Parameters:
          center : (3,) -- a 3D point specifying the sphere's center
          radius : float -- a Python float specifying the sphere's radius
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this sphere.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """
        # Vector from ray origin to sphere center
        oc = ray.origin - self.center

        # Quadratic equation coefficients: at^2 + bt + c = 0
        a = np.dot(ray.direction, ray.direction)  # Should be 1 for normalized direction
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius

        # Compute discriminant
        discriminant = b * b - 4 * a * c

        # No intersection if discriminant is negative
        if discriminant < 0:
            return no_hit

        # Find nearest intersection
        t = (-b - np.sqrt(discriminant)) / (2 * a)

        # Check if nearest intersection is within valid range
        if t < ray.start or t > ray.end:
            # Try the farther intersection
            t = (-b + np.sqrt(discriminant)) / (2 * a)
            if t < ray.start or t > ray.end:
                return no_hit

        # Compute intersection point and normal
        point = ray.origin + t * ray.direction
        normal = (point - self.center) / self.radius

        return Hit(t, point, normal, self.material)


class Triangle:

    def __init__(self, vs, material):
        """Create a triangle from the given vertices.

        Parameters:
          vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
          material : Material -- the material of the surface
        """
        self.vs = vs
        self.material = material

    def intersect(self, ray):
        """Computes the intersection between a ray and this triangle, if it exists.

        Parameters:
          ray : Ray -- the ray to intersect with the triangle
        Return:
          Hit -- the hit data
        """
        a = self.vs[0]
        b = self.vs[1]
        c = self.vs[2]
        p = ray.origin
        d = ray.direction
        A = np.array([a - b, a - c, d]).T
        y = a - p
        x = np.linalg.solve(A, y)
        if x[0] >= 0 and x[1] >= 0 and x[0] + x[1] <= 1:
            t = x[2]
            point = p + t * d
            normal = np.cross(b - a, c - a)
            normal = normal / np.linalg.norm(normal)
            # Check if normal points towards the ray origin
            if np.dot(normal, -d) < 0:
              normal = -normal  # Flip normal if it points the wrong way
            return Hit(t, point, normal, self.material)
        return no_hit

class Disc:
      
      def __init__(self, center, radius = 1, normal = vec([0,1,0]), material = None):
          """Create a Disc with the given center and normal.
  
          Parameters:
            center : (3,) -- a 3D point specifying the Disc's center
            radius : float -- a Python float specifying the Disc's radius
            normal : (3,) -- a 3D unit vector specifying the Disc's normal
            material : Material -- the material of the surface
          """
          self.center = center
          self.radius = radius
          self.normal = normal
          self.material = material
  
      def intersect(self, ray):
          """Computes the intersection between a ray and this Disc, if it exists.
  
          Parameters:
            ray : Ray -- the ray to intersect with the Disc
          Return:
            Hit -- the hit data
          """
          # Compute the denominator of the ray-plane intersection formula
          denom = np.dot(ray.direction, self.normal)
          if np.abs(denom) < 1e-6:
              return no_hit
  
          # Compute the t value of the intersection
          t = np.dot(self.center - ray.origin, self.normal) / denom
          if t < ray.start or t > ray.end:
              return no_hit
          
          point = ray.origin + t * ray.direction

          if np.linalg.norm(point - self.center) > self.radius:
              return no_hit

          return Hit(t, point, self.normal, self.material, self.center)

class Camera:

    def __init__(self, eye=vec([0,0,0]), target=vec([0,0,-1]), up=vec([0,1,0]), 
                 vfov=90.0, aspect=1.0):
        """Create a camera with given viewing parameters.
        Parameters:
          eye : (3,) -- the camera's location, aka viewpoint (a 3D point)
          target : (3,) -- where the camera is looking: a 3D point that appears centered in the view
          up : (3,) -- the camera's orientation: a 3D vector that appears straight up in the view
          vfov : float -- the full vertical field of view in degrees
          aspect : float -- the aspect ratio of the camera's view (ratio of width to height)
        """
        self.eye = eye
        self.aspect = aspect

        # Calculate focal length based on vertical field of view
        self.f = 1.0 / np.tan(np.radians(vfov / 2.0))

        # Calculate viewport dimensions
        self.height = 2.0  # Standard height of 2 units
        self.width = self.height * aspect

        # Compute the camera's coordinate system (u,v,w basis)
        w = eye - target  # W points from target to eye
        w = w / np.linalg.norm(w)

        u = np.cross(up, w)
        u = u / np.linalg.norm(u)

        v = np.cross(w, u)

        # Create the camera-to-world transformation matrix
        rotation = np.column_stack([u, v, w])
        self.M = np.eye(4)
        self.M[:3, :3] = rotation
        self.M[:3, 3] = eye

    def generate_ray(self, img_point):
        """Compute the ray corresponding to a point in the image.

        Parameters:
          img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the upper left
                      corner of the image and (1,1) is the lower right.
        Return:
          Ray -- The ray corresponding to that image location (not necessarily normalized)
        """
        # Transform from texture coordinates [0,1]×[0,1] to image plane coordinates
        # Using the matrix from the screenshot:
        # [w  0  -w/2]
        # [0  -h  h/2]
        # [0  0   1  ]

        x = self.width * img_point[0] - self.width/2
        y = -self.height * img_point[1] + self.height/2

        # Point on image plane at distance f from origin
        image_plane_point = vec([x, y, -self.f])

        # Transform direction to world coordinates using only rotation
        direction = np.dot(self.M[:3, :3], image_plane_point)

        # Normalize the direction
        direction = direction / np.linalg.norm(direction)

        return Ray(self.eye, direction)

# Axis-Aligned Bounding Box
class AABB:
# Each object is wrapped with a bounding box.
# Need a min_point and max_point to represent 2 corners
# to represent a unique bounding box.
    def __init__(self, min_point, max_point):
        self.min_point = np.asarray(min_point, dtype=np.float64)
        self.max_point = np.asarray(max_point, dtype=np.float64)

# check if ray intersects with the bounding box
# efficient, cuz if not even intersecting the box, the entire object
# inside could be skipped. (no_hit)

# Ref: https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection.html
    def intersect(self, ray):
        """Simple ray-box intersection test"""
        inv_dir = np.where(ray.direction != 0, 1.0 / ray.direction, 1e7)
        t1 = (self.min_point - ray.origin) * inv_dir
        t2 = (self.max_point - ray.origin) * inv_dir

        t_min = np.minimum(t1, t2)
        t_max = np.maximum(t1, t2)

        tmin = np.max(t_min)
        tmax = np.min(t_max)

        return tmax >= max(0, tmin)

# Bounding Volume Hierarchy
# Note that this is just like a binary tree
# Ref: https://www.educative.io/answers/what-are-bounding-volume-hierarchies
class BVHNode:
    def __init__(self, triangles, axis=0, depth=0):
        self.left = None
        self.right = None
        self.triangles = triangles
# 1. first, compute a bounding box for all given triangles
        self.bbox = self._compute_bbox(triangles)

        # Stop splitting if we have few triangles or reached max depth
        if len(triangles) <= 4 or depth > 20:
            return

# 2. then, here we split triangles into two groups based on their position
# median is used here for simplicity.
        centroids = np.array([np.mean(tri.vs, axis=0) for tri in triangles])
        median_idx = len(triangles) // 2
        sorted_indices = np.argsort(centroids[:, axis])

        # Split triangles into left and right groups
        left_tris = [triangles[i] for i in sorted_indices[:median_idx]]
        right_tris = [triangles[i] for i in sorted_indices[median_idx:]]

        # Only create child nodes if we split the triangles
        if left_tris and right_tris:
# split around x,y,z,x,y,z.... to ensure good spatial division
            next_axis = (axis + 1) % 3
# 3. after that, recursively create left_child and right_child
            self.left = BVHNode(left_tris, next_axis, depth + 1)
            self.right = BVHNode(right_tris, next_axis, depth + 1)
            # Clear triangles list for child nodes
            self.triangles = None

    def _compute_bbox(self, triangles):
        """Compute bounding box for a list of triangles"""
        if not triangles:
            return AABB(np.zeros(3), np.zeros(3))

        vertices = np.vstack([tri.vs for tri in triangles])
        return AABB(np.min(vertices, axis=0), np.max(vertices, axis=0))

    def intersect(self, ray):
        """Find the closest intersection with triangles in this BVH node"""
# 1. first, check if ray intersects the bounding box
        if not self.bbox.intersect(ray):
            return no_hit

# 2. then, if leaf node, test all triangles in the current bbox.
        if self.triangles is not None:
            closest_hit = no_hit
            for triangle in self.triangles:
                hit = triangle.intersect(ray)
                if ray.start <= hit.t < min(ray.end, closest_hit.t):
                    closest_hit = hit
            return closest_hit

# 3. if child node, then it has no triangles itself as init above
# need to check both children
        closest_hit = no_hit

        # Check left child
        if self.left:
            left_hit = self.left.intersect(ray)
            if left_hit.t < closest_hit.t:
                closest_hit = left_hit
                ray.end = min(ray.end, closest_hit.t)

        # Check right child
        if self.right:
            right_hit = self.right.intersect(ray)
            if right_hit.t < closest_hit.t:
                closest_hit = right_hit

        return closest_hit

class PointLight:

    def __init__(self, position, intensity):
        """Create a point light at given position and with given intensity

        Parameters:
          position : (3,) -- 3D point giving the light source location in scene
          intensity : (3,) or float -- RGB or scalar intensity of the source
        """
        self.position = position
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """

        dircectVec = self.position - hit.point
        distance = np.linalg.norm(dircectVec)
        direction = dircectVec / distance


        lightRay = Ray(self.position, -direction)
        firstHit = scene.intersect(lightRay)
        # If the light is blocked by another object, return black
        if firstHit.t < distance - 1e-5:
            return vec([0,0,0])

        ray_direction = -ray.direction
        h = direction + ray_direction
        h = h / np.linalg.norm(h)
        
        cosine = np.dot(direction, hit.normal)

        light = 0
        if cosine >= 0:
            diffusion = hit.material.k_d * cosine
            specular = hit.material.k_s * (np.dot(hit.normal, h) ** hit.material.p) * cosine
            light = cosine * self.intensity / (distance ** 2) * (diffusion + specular)
        return light


class AmbientLight:

    def __init__(self, intensity):
        """Create an ambient light of given intensity

        Parameters:
          intensity (3,) or float: the intensity of the ambient light
        """
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        return self.intensity * hit.material.k_a


class Scene:

    def __init__(self, surfs, bg_color=vec([0.53, 0.8, 0.92])):
        """Create a scene containing the given objects.

        Parameters:
          surfs : [Sphere, Triangle] -- list of the surfaces in the scene
          bg_color : (3,) -- RGB color that is seen where no objects appear
        """
        self.surfs = surfs
        self.bg_color = bg_color

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """
        # `no-hit` -> Value to represent absence of an intersection
        closest_hit = no_hit
        for surface in self.surfs:
            hit = surface.intersect(ray)
            if hit.t < closest_hit.t:
                closest_hit = hit
        return closest_hit

class AcceleratedScene(Scene):
    def __init__(self, surfs, bg_color=vec([0.53, 0.8, 0.92])):
        # get surfaces and bg_color
        super().__init__(surfs, bg_color)

        # separate triangles and other surfaces, if any.
        self.triangles = [surf for surf in surfs if isinstance(surf, Triangle)]
        self.other_surfs = [surf for surf in surfs if not isinstance(surf, Triangle)]

        # build BVH if there are triangles
        self.bvh = BVHNode(self.triangles) if self.triangles else None

        print(f"Scene initialized with {len(self.triangles)} triangles and {len(self.other_surfs)} other surfaces")

    def intersect(self, ray):
        closest_hit = no_hit

        # prioritize BVH for ray-triangle acceleration
        if self.bvh:
            closest_hit = self.bvh.intersect(ray)

        # then simply check other surfaces using old way
        for surf in self.other_surfs:
            hit = surf.intersect(ray)
            if hit.t < closest_hit.t:
                closest_hit = hit

        return closest_hit

def verify_triangles(scene, camera, nx=10, ny=10):
    """Test a few rays to verify triangle intersections"""
    print("Verifying triangle intersections...")
    hits = 0
    for i in range(ny):
        for j in range(nx):
            img_point = ((j+0.5)/nx, (i+0.5)/ny)
            ray = camera.generate_ray(img_point)
            hit = scene.intersect(ray)
            if hit.t < np.inf:
                hits += 1
    print(f"Found {hits} intersections in test grid")

MAX_DEPTH = 4

def diffusion(pointLight, ambientLight, material):
    return pointLight * material.k_d + ambientLight * material.k_a

def specular(pointLight, ray, hit):
    return hit.material.k_s

def shade(ray, hit, scene, lights, depth=0):
    """Compute shading for a ray-surface intersection.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene
      lights : [PointLight or AmbientLight] -- the lights
      depth : int -- the recursion depth so far
    Return:
      (3,) -- the color seen along this ray
    When mirror reflection is being computed, recursion will only proceed to a depth
    of MAX_DEPTH, with zero contribution beyond that depth.
    """
    if hit.t == np.inf:
        return scene.bg_color  # No intersection, return background color

    if depth == MAX_DEPTH:
        return vec([0,0,0])

    total_light = vec([0,0,0])    
    material = hit.material
    # Fresnel factor
    R0 = material.R0
    normal = hit.normal

    # Mirror reflection
    # mirrorReflection = vec([0,0,0])
    # if depth < MAX_DEPTH:
    #     normal = hit.normal
    #     v = -ray.direction
    #     r = 2 * normal * np.dot(v, normal) - v
    #     new_ray = Ray(hit.point, r, 1e-5)
    #     mirrorReflection = shade(new_ray, scene.intersect(new_ray), scene, lights, depth + 1) * hit.material.k_m
        
    # Refraction and Fresnel reflection, only condider interact with air
    cos_theta1 = np.clip(np.dot(ray.direction, hit.normal), -1, 1)
    abs_cos_theta1 = np.abs(cos_theta1)
    R = R0 + (1 - R0) * (1 - abs_cos_theta1) ** 5
    v = -ray.direction
    r = 2 * normal * np.dot(v, normal) - v
    reflectSrcRay = Ray(hit.point, r, 1e-5)

    if hit.material.n < 9.9: # if material is transparent
        if cos_theta1 <= 0: # ray is outside the object
            n1 = 1
            n2 = material.n
            cos_theta1 = -cos_theta1
            sin_theta1 = np.sqrt(1 - cos_theta1 ** 2)
            sin_theta2 = n1 * sin_theta1 / n2
            cos_theta2 = np.sqrt(1 - sin_theta2 ** 2)
            refraction_direction = (ray.direction + cos_theta1 * normal) / sin_theta1 * sin_theta2 - normal * cos_theta2
            refraction_direction = refraction_direction / np.linalg.norm(refraction_direction)
            refraction_ray = Ray(hit.point, refraction_direction, 1e-5)
            total_light += (1 - R) * material.k_t * shade(refraction_ray, scene.intersect(refraction_ray), scene, lights, depth + 1) #refraction
        else: # ray is inside the object
            n1 = material.n
            n2 = 1
            sin_theta1 = np.sqrt(1 - cos_theta1 ** 2)
            sin_theta2 = n1 * sin_theta1 / n2
            assert(cos_theta1 <= 1)
            if sin_theta2 < 1:
                cos_theta2 = np.sqrt(1 - sin_theta2 ** 2)
                refraction_direction = (ray.direction - cos_theta1 * hit.normal) / sin_theta1 * sin_theta2 + normal * cos_theta2
                refraction_direction = refraction_direction / np.linalg.norm(refraction_direction)
                refraction_ray = Ray(hit.point, refraction_direction, 1e-5)
                total_light += (1 - R) * material.k_t * shade(refraction_ray, scene.intersect(refraction_ray), scene, lights, depth + 1) #refraction
            else:
                R = 1
    # fresnel reflection
    total_light += R * shade(reflectSrcRay, scene.intersect(reflectSrcRay), scene, lights, depth + 1) * material.k_m

    # slim lens simulation
    if hit.material.f is not None:
        f = hit.material.f
        obj = ray.origin
        opticalcenter = hit.opticalcenter
        zo = np.abs(np.dot(obj - opticalcenter, hit.normal))
        bias = (np.linalg.norm(obj - opticalcenter)/zo)**2 - 1
        zi = 1 / (1 / f - 1 / zo - 0.1 * bias - 0.01 * bias ** 2)
        img = opticalcenter + (opticalcenter - obj) * zi / zo
        if zi > 0: # convex lens
            outRayDirect = img - hit.point
        else: # concave lens
            outRayDirect = hit.point - img
        outRayDirect = outRayDirect / np.linalg.norm(outRayDirect)
        outRay = Ray(hit.point, outRayDirect, 1e-5)
        total_light += material.k_t * shade(outRay, scene.intersect(outRay), scene, lights, depth + 1)


    for light in lights:
        if isinstance(light, AmbientLight):
            total_light += light.illuminate(ray, hit, scene) # ambient light
        if isinstance(light, PointLight):
            total_light += light.illuminate(ray, hit, scene) # point light specular and diffuse
    return total_light

def render_image(camera, scene, lights, nx, ny):
    """Render a ray traced image.

    Parameters:
      camera : Camera -- the camera defining the view
      scene : Scene -- the scene to be rendered
      lights : Lights -- the lights illuminating the scene
      nx, ny : int -- the dimensions of the rendered image
    Returns:
      (ny, nx, 3) float32 -- the RGB image
    """
    output_image = np.zeros((ny,nx,3), np.float32)

    # Iterate over each pixel in the image
    for i in range(ny):
        for j in range(nx):
            img_point = ((j+0.5) / nx, (i+0.5) / ny)
            # Generate ray for this pixel using the camera
            ray = camera.generate_ray(img_point)
            intersection = scene.intersect(ray)
            # Compute the color at the intersection point or background color if no hit
            color = shade(ray, intersection, scene, lights)
            # Set the output pixel color
            output_image[i, j] = color

    return output_image

def render_image_with_progress(camera, scene, lights, nx, ny):
    """Render a ray traced image with progress tracking.

    Parameters:
      camera : Camera -- the camera defining the view
      scene : Scene -- the scene to be rendered
      lights : Lights -- the lights illuminating the scene
      nx, ny : int -- the dimensions of the rendered image
    Returns:
      (ny, nx, 3) float32 -- the RGB image
    """
    output_image = np.zeros((ny,nx,3), np.float32)
    total_pixels = nx * ny
    pixels_done = 0
    start_time = time.time()
    last_update = start_time
    update_interval = 1.0  # Update progress every second

    def estimate_time_remaining(pixels_done, total_pixels, elapsed_time):
        if pixels_done == 0:
            return "calculating..."
        pixels_per_second = pixels_done / elapsed_time
        remaining_seconds = (total_pixels - pixels_done) / pixels_per_second
        return str(timedelta(seconds=int(remaining_seconds)))

    # Iterate over each pixel in the image
    for i in range(ny):
        for j in range(nx):
            img_point = ((j+0.5) / nx, (i+0.5) / ny)
            # Generate ray for this pixel using the camera
            ray = camera.generate_ray(img_point)
            intersection = scene.intersect(ray)
            # Compute the color at the intersection point or background color if no hit
            color = shade(ray, intersection, scene, lights)
            # Set the output pixel color
            output_image[i, j] = color
            
            # Update progress
            pixels_done += 1
            current_time = time.time()
            if current_time - last_update >= update_interval:
                elapsed = current_time - start_time
                progress = (pixels_done / total_pixels) * 100
                eta = estimate_time_remaining(pixels_done, total_pixels, elapsed)
                print(f"\rProgress: {progress:.1f}% | "
                      f"Pixels: {pixels_done}/{total_pixels} | "
                      f"Elapsed: {str(timedelta(seconds=int(elapsed)))} | "
                      f"ETA: {eta}", end="")
                last_update = current_time
    
    # Print final progress
    total_time = time.time() - start_time
    print(f"\nRendering completed in {str(timedelta(seconds=int(total_time)))}")
    return output_image
