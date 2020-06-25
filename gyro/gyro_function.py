import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import torch
import torchgeometry as tgm
from torch.autograd import Variable

# Quaternion: [x, y, z, w]

def norm_quat(quat):
    norm_quat = LA.norm(quat)   
    if norm_quat > 1e-6:
        quat = quat / norm_quat   
        #     [0 norm_quat norm_quat - 1e-6]
    else:
        # print('bad len for Reciprocal')
        quat = np.array([0,0,0,1])
    return quat

def diff(data1):
    data2 = np.loadtxt("/home/zhmeishi_google_com/dvs/data/testdata/matlab/compare_data.txt")
    print(np.sum(np.abs(data1 - data2)))
    print(np.sum(np.abs(data1) + np.abs(data2)))
    print(np.min(data1))
    print(np.min(data2))
    
def ConvertAxisAngleToQuaternion(axis, angle):
    axis = axis/LA.norm(axis)  
    half_angle = angle*0.5  
    sin_half_angle = np.sin(half_angle)
    quat = np.array([sin_half_angle* axis[0], sin_half_angle* axis[1], sin_half_angle* axis[2], np.cos(half_angle)])

    return norm_quat(quat)

def AngularVelocityToQuat(angular_v, dt):
    length = LA.norm(angular_v)  
    if length < 1e-6:
        angular_v = np.array([1, 0, 0])  
        print('bad length')
    else:
        angular_v = angular_v/length  
    quat = ConvertAxisAngleToQuaternion(angular_v, length*dt) 
    return quat

def QuaternionProduct(q1, q2):
    x1 = q1[0]  
    y1 = q1[1]   
    z1 = q1[2]   
    w1 = q1[3]   

    x2 = q2[0]  
    y2 = q2[1]  
    z2 = q2[2]  
    w2 = q2[3]  

    quat = np.zeros(4)
    quat[3] =  w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  
    quat[0] =  w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  
    quat[1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  
    quat[2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2 

    return norm_quat(quat)

def torch_QuaternionProduct(q1, q2, USE_CUDA = True):
    x1 = q1[:,0]  
    y1 = q1[:,1]   
    z1 = q1[:,2]   
    w1 = q1[:,3]   

    x2 = q2[:,0]  
    y2 = q2[:,1]  
    z2 = q2[:,2]  
    w2 = q2[:,3]  

    batch_size = q1.size()[0]
    quat = Variable(torch.zeros((batch_size, 4), requires_grad=True))
    # quat_out = Variable(torch.zeros((batch_size, 4), requires_grad=True))
    if USE_CUDA == True:
        quat = quat.cuda()
    
    quat[:,3] =  w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  
    quat[:,0] =  w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  
    quat[:,1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  
    quat[:,2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  

    # for i in range(batch_size):
    #     norm_quat = torch.norm(quat[i])   
    #     if norm_quat > 1e-6:        
    #         quat_out[i] = quat[i] / norm_quat   
    #         #     [0 norm_quat norm_quat - 1e-6]
    #     else:
    #         quat_out[i] = quat[i] * 0
    #         quat_out[i, 3] = 1
    quat = quat / (torch.unsqueeze(torch.norm(quat, dim = 1), 1) + 1e-6)
    return quat

def ProcessGyroRotation(gyro_data):
    num_inputs = np.shape(gyro_data)[0]
    quats = np.zeros((num_inputs, 4))  
    quats[0,:] = np.array([0, 0, 0, 1])
    for i in range(1, num_inputs):
        dt = (gyro_data[i, 0] - gyro_data[i-1, 0])*1e-9  
        quat = AngularVelocityToQuat(gyro_data[i, 1:4], dt)  
        quats[i,:] = QuaternionProduct(quat, quats[i-1,:])   # R_t = delta R_t * R_t-1
        quats[i,:] = quats[i,:] / LA.norm(quats[i,:]) 
    return quats 

def QuaternionReciprocal(q):
    quat = np.array([-q[0], -q[1], -q[2], q[3]])  
    return norm_quat(quat)

def torch_QuaternionReciprocal(q,  USE_CUDA = True):
    quat = torch.cat((-q[:,0:1], -q[:,1:2], -q[:,2:3], q[:,3:]), dim = 1) 
    batch_size = quat.size()[0]

    # for i in range(batch_size):
    #     norm_quat = torch.norm(quat[i])   
    #     if norm_quat > 1e-6:        
    #         quat[i] = quat[i] / norm_quat   
    #         #     [0 norm_quat norm_quat - 1e-6]
    #     else:
    #         quat[i] = quat[i] * 0
    #         quat[i, 3] = 1
    quat = quat / (torch.unsqueeze(torch.norm(quat, dim = 1), 1) + 1e-6)
    return quat

def ConvertQuaternionToAxisAngle(quat):
    quat = quat/LA.norm(quat)   
    axis_norm = LA.norm(quat[0:3])
    axis = np.array([1.0, 0.0, 0.0])
    if axis_norm < 1e-6:
        angle = 0   
        #     [axis_norm 1e-6]
    else:
        axis_norm_reciprocal = 1/axis_norm   
        axis[0] = quat[0] * axis_norm_reciprocal   
        axis[1] = quat[1] * axis_norm_reciprocal   
        axis[2] = quat[2] * axis_norm_reciprocal   
        angle = 2 * np.arccos(quat[3])
    return [axis, angle]

def ProcessGyroData(gyro_data):
    quats = ProcessGyroRotation(gyro_data) 
    size = np.shape(gyro_data)[0]
    axis_dif = np.zeros((size,3)) 
    for i in range(1, size):
        quat_dif = QuaternionProduct(quats[i,:], QuaternionReciprocal(quats[i-1,:]))  
        [axis_dif_cur, angles_cur] = ConvertQuaternionToAxisAngle(quat_dif)  
        axis_dif[i,:] = axis_dif_cur*angles_cur  
    return [axis_dif, quats]


def SlerpWithDefault(q1, q2, t, q_default):
    t = max(min(t, 1.0), 0.0) 
    kEpsilon = 1e-6 
    kSlerpLinearThresh = 0.9995 
    
    q1 = q1/LA.norm(q1) 
    q2 = q2/LA.norm(q2) 

    if t < kEpsilon:
        q3 = q1 
        return q3
    elif t > 1-kEpsilon:
        q3 = q2 
        return q3

    dot_prodcut = np.sum(q1*q2) 

    if abs(dot_prodcut) >= 1:
        q3= q_default 
        return q3
    elif abs(dot_prodcut) > kSlerpLinearThresh:
        q3 = q1*(1-t) + q2*t 
        q3 = q3/LA.norm(q3)     
        return q3

    sign = 1 
    if dot_prodcut < 0:
        sign = -1 
        dot_prodcut = -dot_prodcut 

    theta = np.arccos(dot_prodcut) 
    sin_theta = np.sin(theta) 
    inv_sin_theta = 1.0 / sin_theta 
    coeff1 = np.sin((1.0 - t) * theta) * inv_sin_theta 
    coeff2 = sign * np.sin(t * theta) * inv_sin_theta 
    q3 = q1 * coeff1 + q2 * coeff2 
    return q3


def GetGyroAtTimeStamp(gyro_data, timestamp):
    z = np.array([0,0,0,1]) 
    if len(gyro_data) >= 2 and (not(timestamp < gyro_data[0,0] or timestamp > gyro_data[-1, 0])):
        ind = np.where(gyro_data[:,0] >= timestamp)
        ind = np.squeeze( ind, axis = 0)
        if gyro_data[ind[0], 0] == timestamp:
            z = gyro_data[ind[0],1:]
        else:
            start_index = ind[0] -1 
            end_index = ind[0] 
            ratio = (timestamp - gyro_data[start_index,0])/(gyro_data[end_index,0]-gyro_data[start_index,0])
            z = SlerpWithDefault(gyro_data[start_index,1:], gyro_data[end_index, 1:], ratio, gyro_data[start_index,1:]) 
    z = z / (LA.norm(z) + 1e-6)
    return z

def FindOISAtTimeStamp(ois_log, time):
    ois_time = ois_log[:,2] 
    if time <= ois_time[0]:
        ois_data = ois_log[0, 0:2] 
    elif time > ois_time[-1]:
        ois_data = ois_log[-1, 0:2]
    else:
        ind = np.where(ois_time >= time)
        ind = np.squeeze( ind, axis = 0)
        first_ind = ind[0]
        if ois_time[first_ind] == ind[0]:
            ois_data = ois_log[first_ind, 0:2]
        else:
            cur_time = ois_time[first_ind] 
            last_timestamp = ois_time[first_ind - 1]
            ratio = (time - last_timestamp) / (cur_time - last_timestamp) 
            ois_data = ois_log[first_ind - 1,0:2] * (1-ratio) + ois_log[first_ind,0:2]*ratio 

    return ois_data

def GetMetadata(frame_data, frame_index, result_poses = {} ):
    # global static_options
    # We can just use 1.27 as fov and virtual fov for videos in the data set.
    metadata = {}
    metadata["frame_id"] = frame_index
    metadata["timestamp_ns"]  = frame_data[frame_index, 0]
    metadata["timestamp_ois_ns"]  = frame_data[frame_index, 4]
    metadata["rs_time_ns"]  = frame_data[frame_index, 3]
    if "real fov" in result_poses:
        metadata["fov"] = result_poses['real fov'][frame_index,:] 
    else:
        metadata["fov"] = 1.27
    if "virtual fov" in result_poses:
        metadata["virtual_fov"] = result_poses['virtual fov'][frame_index,:] 
    else:
        metadata["virtual_fov"] = 1.27

    return metadata

def GetProjections(static_options, metadata, quats_data, ois_data):
    num_rows = static_options["num_grid_rows"]
    real_projections = []
    for i in range(num_rows):
        timestmap_ns = metadata["timestamp_ns"] + metadata["rs_time_ns"] * i / (num_rows-1)
        timestamp_ois_ns = metadata["timestamp_ois_ns"] + metadata["rs_time_ns"] * i / (num_rows-1)
        real_projections.append(GetRealProjection(
            static_options, quats_data, ois_data, metadata["fov"], timestmap_ns, timestamp_ois_ns))
    return real_projections

def GetRealProjection(static_options, quats_data, ois_data, fov, timestamp_ns, timestamp_ois_ns):
    quat = GetGyroAtTimeStamp(quats_data, timestamp_ns)
    ois_offset = FindOISAtTimeStamp(ois_data, timestamp_ois_ns) 
    # ois is w.r.t. active array size, thus we need to convert it to normalzied space.

    ois_offset = np.array(ois_offset) / np.array([static_options["crop_window_width"], static_options["crop_window_height"]])
    projection = GetProjectionHomography(quat, fov, ois_offset, static_options["width"], static_options["height"])
    return projection

def GetProjectionHomography(rot, fov, offset, width, height):
    # rot: rotation in quaternion
    # fov: sensor_width / focal_length.
    # offset: additional ois offset at normalized domain.
    # width/height: frame size.
    focal_length = width / fov
    rotation = ConvertQuaternionToRotationMatrix(rot)
    intrinsics = GetIntrinsics(focal_length, offset, width, height)
    projection_homography = np.matmul(intrinsics, rotation)
    return projection_homography

def ConvertQuaternionToRotationMatrix(quat):
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]
    rotation = np.zeros(9)
    rotation[0] = 1 - 2 * y * y - 2 * z * z
    rotation[1] = 2 * x * y - 2 * z * w
    rotation[2] = 2 * x * z + 2 * y * w
    rotation[3] = 2 * x * y + 2 * z * w
    rotation[4] = 1 - 2 * x * x - 2 * z * z
    rotation[5] = 2 * y * z - 2 * x * w
    rotation[6] = 2 * x * z - 2 * y * w
    rotation[7] = 2 * y * z + 2 * x * w
    rotation[8] = 1 - 2 * x * x - 2 * y * y
    rotation = np.reshape(rotation, (3, 3)) # Note reshape is different with matlab
    return rotation

def torch_ConvertQuaternionToRotationMatrix(quat, USE_CUDA = True):
    x = quat[:,0]
    y = quat[:,1]
    z = quat[:,2]
    w = quat[:,3]

    batch_size = quat.size()[0]
    rotation = Variable(torch.zeros((batch_size, 9), requires_grad=True))
    if USE_CUDA == True:
        rotation = rotation.cuda()

    rotation[:,0] = 1 - 2 * y * y - 2 * z * z
    rotation[:,1] = 2 * x * y - 2 * z * w
    rotation[:,2] = 2 * x * z + 2 * y * w
    rotation[:,3] = 2 * x * y + 2 * z * w
    rotation[:,4] = 1 - 2 * x * x - 2 * z * z
    rotation[:,5] = 2 * y * z - 2 * x * w
    rotation[:,6] = 2 * x * z - 2 * y * w
    rotation[:,7] = 2 * y * z + 2 * x * w
    rotation[:,8] = 1 - 2 * x * x - 2 * y * y
    rotation = rotation.view(batch_size, 3, 3) # Note reshape is different with matlab
    return rotation

def ConvertRotationMatrixToQuaternion(m):
    tr = m[0,0] + m[1,1] + m[2,2]
    if tr > 0 :
        S = 2 * (tr+1.0)**0.5
        qw = 0.25 * S
        qx = (m[2,1] - m[1,2]) / S
        qy = (m[0,2] - m[2,0]) / S
        qz = (m[1,0] - m[0,1]) / S
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        S = 2* (1.0 + m[0,0] - m[1,1] - m[2,2]) ** 0.5
        qw = (m[2,1] - m[1,2]) / S
        qx = 0.25 * S
        qy = (m[0,1] + m[1,0]) / S
        qz = (m[0,2] + m[2,0]) / S
    elif m[1,1] > m[2,2]: 
        S = 2* (1.0 - m[0,0] + m[1,1] - m[2,2]) ** 0.5
        qw = (m[0,2] - m[2,0]) / S
        qx = (m[0,1] + m[1,0]) / S
        qy = 0.25 * S
        qz = (m[1,2] + m[2,1]) / S
    else: 
        S = 2* (1.0 - m[0,0] - m[1,1] + m[2,2]) ** 0.5
        qw = (m[1,0] - m[0,1]) / S
        qx = (m[0,2] + m[2,0]) / S
        qy = (m[1,2] + m[2,1]) / S
        qz = 0.25 * S
    return np.array([qx,qy,qz,qw])

def torch_ConvertRotationMatrixToQuaternion(m, USE_CUDA = True):
    batch_size = m.size()[0]
    res = torch.unsqueeze(torch.Tensor([[0,0,0]]).repeat(batch_size,1),2)
    if USE_CUDA == True:
        res = res.cuda()

    matrix3d = torch.cat((m,res), dim = 2)
    q = tgm.rotation_matrix_to_quaternion(matrix3d)  # Still need to consider diff betweem [0,0,0,0] and [0,0,0,1]
    q = torch.cat((q[:,1:],q[:,:1]), dim =1)
    return q

def GetIntrinsics(focal_length, offset, width, height):
    intrinsics = [
        [float(focal_length), 0.0, 0.5*(width-1)+offset[0]*width], 
        [0.0, float(focal_length), 0.5*(height-1)+offset[1]*height], 
        [0.0, 0.0, 1.0]
        ]
    return np.array(intrinsics)


def GetVirtualProjection(static_options, result_pose, metadata, frame_index):
    # debug only, for getting results and references for comparisons.
    quat = result_pose['virtual pose'][frame_index,:]
    if 'vitual lens offset' in result_pose:
        virutal_lens_offset = result_pose['vitual lens offset'][frame_index,:]
    else:
        virutal_lens_offset = np.array([0,0])
    virtual_projection = GetProjectionHomography(
        quat, metadata["virtual_fov"], virutal_lens_offset, static_options["width"], static_options["height"])
    return virtual_projection


def GetForwardGrid(static_options, real_projections, virtual_projection):
    # real_projections: a set of 3x3 projections.
    # virtual_projection: a single 3x3 projection.

    grid = np.zeros((4, static_options["num_grid_cols"], static_options["num_grid_rows"]))
    width = static_options["width"]
    height = static_options["height"]

    row_step = 1/ (static_options["num_grid_rows"] - 1)
    col_step = 1/ (static_options["num_grid_cols"] - 1)

    for i in range(static_options["num_grid_rows"]):
        transform = GetHomographyTransformFromProjections(real_projections[i], virtual_projection)
        v = i * row_step
        for j in range(static_options["num_grid_cols"]):
            u = j * col_step
            point = np.array([u * width, v * height, 1]).T
            warped_point = ApplyTransform(transform, point)
            warped_point = warped_point / np.array([width, height, 1]) # normalize
            grid[:, j, i] = np.array([warped_point[0], warped_point[1], u, v])
    return grid

def GetHomographyTransformFromProjections(proj_src, proj_dst):
    return np.matmul(proj_dst, LA.inv(proj_src))

def ApplyTransform(transform, point):
    # Warps a 2D point ([x y 1]) using a homography transform.
    # Returns the warped 2D point ([warped_x, warped_y, 1]).
    z = np.matmul(transform, point)
    z = z / z[2]
    return z

def CenterZoom(grid, ratio):
    grid[:, 0:2, :, :]  = (grid[:, 0:2, :, :] - 0.5) * ratio + 0.5
    return grid

def GetWarpingFlow(real_projections_src, real_projections_dst, num_rows, num_cols, frame_width, frame_height):
    # num_rows: rows of the flow.
    # num_cols: cols of the flow.
    grid = np.zeros((4, num_cols, num_rows))

    row_step = 1/ (num_rows - 1)
    col_step = 1/ (num_cols - 1)

    for i in range(num_rows):
        transform = GetHomographyTransformFromProjections(real_projections_src[i], real_projections_dst[i])
        v = i * row_step
        for j in range(num_cols):
            u = j * col_step
            point = np.array([u * frame_width, v * frame_height, 1]).T
            warped_point = ApplyTransform(transform, point)
            warped_point = warped_point / np.array([frame_width, frame_height, 1]) # normalize
            grid[:, j, i] = np.array([warped_point[0], warped_point[1], u, v])
    return grid


if __name__ == "__main__":
    q = torch.Tensor([[1,2,113,14],[1,2,113,15],[0,0,0,0]]).cuda()
    m = torch_ConvertQuaternionToRotationMatrix(q)
    q1 = torch_ConvertRotationMatrixToQuaternion(m)
    r = torch_QuaternionReciprocal(q)
    print(r)