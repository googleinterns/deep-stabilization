import os
import torch
import time
import numpy as np 
import sys
import cv2
import math
# Util function for loading meshes
# from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,  
    DirectionalLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,
    TexturedSoftPhongShader
)


# Setup
device = torch.device("cuda")
# torch.cuda.set_device(device)

def process_grids(grids):
    rows = grids[0]["vertex_grid_rows"]
    cols = grids[0]["vertex_grid_cols"]
    num_points = rows * cols

    length = len(grids)
    verts = torch.zeros((length,num_points,3), dtype = torch.float)
    verts_uvs = torch.zeros((length,num_points,2), dtype = torch.float)
    for i in range(length):
        grid = torch.Tensor(grids[i]["warping grid"]).view(num_points, 4)
        verts[i,:,:2] = grid[:,:2]
        verts_uvs[i,:,:] = grid[:,2:]
    verts = (verts * 2 - 1)   

    traingle_index = []
    for i in range((rows-1)*cols):
        if i % cols == cols - 1:
            continue
        traingle_index.append(i)
    traingle_index = np.array(traingle_index)
    faces_upper = torch.tensor(np.array([traingle_index,traingle_index+1,traingle_index+cols]).T, dtype = torch.int64)
    faces_lower = torch.tensor(np.array([traingle_index+1,traingle_index+cols+1,traingle_index+cols]).T, dtype = torch.int64)
    faces = torch.cat((faces_upper,faces_lower), dim = 0)
    faces = faces.repeat(length, 1, 1)
    return verts.to(device), verts_uvs.to(device), faces.to(device)

def rendering_batch(renderer, grids, images, size):
    verts, verts_uvs, faces = process_grids(grids)
    images = np.array(images)
    images = torch.tensor(images,dtype=torch.float)/255
    images = images.to(device)

    tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces, maps=images)
    mesh = Meshes(verts=verts, faces=faces, textures=tex)

    t1 = time.time()
    images = renderer(mesh)
    t2 = time.time()
    print(t2 - t1)
    images = torch.clamp(images[:,:,:,:3]*255, min=0, max=255).cpu().numpy().astype("uint8")
    # images = torch.nn.functional.upsample(images, size=(size[1],size[0]), mode='bilinear')
    return images

def rendering(grids, frame_array, size, batch = 20):
    # Initialize an OpenGL perspective camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
    R, T = look_at_view_transform(0.932, 0, 0) # 3**0.5, 0.932
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=max(size), 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
    # -z direction. 
    lights = DirectionalLights(device=device, ambient_color=[[1.0, 1.0, 1.0]],
            diffuse_color=[[0.0, 0.0, 0.0]],specular_color=[[0.0, 0.0, 0.0]],direction=[[0.0, 0.0, 1.0]])

    # Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=TexturedSoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    
    images = None
    for i in range(math.ceil(len(grids)/batch)):
        if images is None:
            images = rendering_batch(renderer, grids[i*batch:(i+1)*batch],frame_array[i*batch:(i+1)*batch],size)
        else:
            out = rendering_batch(renderer, grids[i*batch:(i+1)*batch],frame_array[i*batch:(i+1)*batch],size)
            images = np.concatenate((images, out), axis = 0)
        
    imgs = []
    for i in range(images.shape[0]):
        imgs.append(cv2.resize(images[i], size, interpolation = cv2.INTER_LINEAR))
    return imgs