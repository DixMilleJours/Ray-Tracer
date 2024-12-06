# Hi there!
This is a fully functional object renderer (shadowing, mirror reflection, refraction, material, specular shading, etc.)
# Background
- With my previous project, _Funland_, I got to build an immersive VR game with Unity3D. With an interest to delve into how computer graphics are rendered under the hood, I implemented this project during the course on _Computer Graphics_.
# Intro
- This object render implements various key features to nicely render objects created in Blender.
# Features
1. Accelerated Triangle-Ray intersection:
![scene-1](https://github.com/user-attachments/assets/e67b2a37-ce1e-44cb-9011-272f9ba09c10)
- I leveraged Axis-Aligned Bounding Box and Bounding Volume Hierarchy (like a binary tree) to implement accelerated rendering.
2. Fresnel reflection and refraction, convex & concave lensing effect:
![final](https://github.com/user-attachments/assets/94bf709b-cd62-4d4a-9a7d-0b577c3bee3c)
- As shown in the rendered image, we could see the convex and concave lensing effect. Moreover, the sphere demonstrates the Fresnel reflection and refraction effect.
3. Mirror reflection, specular shading, shadowing:
![mirror](https://github.com/user-attachments/assets/4cd92b19-6334-4c4e-9530-c2da6640af91)
- As shown above, we could clearly see the mirror reflection effect. Moreover, the spheres have realistic specular shading and shadowing.
