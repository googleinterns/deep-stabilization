import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from gyro import (
diff,
GetGyroAtTimeStamp,
QuaternionProduct,
QuaternionReciprocal,
ConvertQuaternionToAxisAngle,
FindOISAtTimeStamp,
GetMetadata,
GetProjections,
GetVirtualProjection,
GetForwardGrid,
CenterZoom,
GetWarpingFlow
)
from gyro import LoadOISData, LoadStabResult, LoadFrameData, LoadGyroData, get_static


def GyroPlotDemo(gyro_log_name, ois_log_name, frame_log_name):
    # Process frame metadata.
    frame_data = LoadFrameData(frame_log_name)
    
    # Process gyro data.
    quats_data = LoadGyroData(gyro_log_name) 

    # Load OIS data.
    ois_data = LoadOISData(ois_log_name) 

    # Obtain per-frame gyro pose.
    num_frames = np.shape(frame_data)[0] 
    quats = np.zeros((num_frames, 4)) 
    lens_offsets = np.zeros((num_frames, 2)) 
    for i in range(num_frames):
        quats[i,:] = GetGyroAtTimeStamp(quats_data, frame_data[i,0]) 

    # Visualize them.
    rotations = np.zeros((num_frames,3))
    for i in range(num_frames):
        if i != 0:
            quat_dif = QuaternionProduct(quats[i,:], QuaternionReciprocal(quats[i-1,:])) 
            [axis_dif_cur, angles_cur] = ConvertQuaternionToAxisAngle(quat_dif) 
            rotations[i,:] = axis_dif_cur*angles_cur 
        lens_offsets[i,:] = FindOISAtTimeStamp(ois_data, frame_data[i, 4])     

    # figure('units','normalized','outerposition',[0 0 1 1])
    plt.subplot(5,1,1)
    plt.plot(rotations[:,0])
    plt.xlabel('gyro x')

    plt.subplot(5,1,2)
    plt.plot(rotations[:,1])
    plt.xlabel('gyro y')

    plt.subplot(5,1,3)
    plt.plot(rotations[:,2])
    plt.xlabel('gyro z')
    
    plt.subplot(5,1,4)
    plt.plot(lens_offsets[:,0])
    plt.xlabel('ois x')

    plt.subplot(5,1,5)
    plt.plot(lens_offsets[:,1])
    plt.xlabel('ois y')

    plt.savefig("./python.jpg")
    return


def EndToEndDemo(gyro_log_name, ois_log_name, frame_log_name, result_poses_name):
    # This demo shows end to end process to generate a forward warping grid
    # from input gyro/ois/frame metadata logs, and a resulting pose. The
    # resulting pose can be obtained by traditional filtering, or inferred from
    # CNN, obtained by LoadStabResult.m.
    # A pose is represented as (rotation, offset).
    static_options = get_static()

    # A matadata has the following fields.
    # frame_id
    # timestamp_ns
    # timestamp_ois_ns
    # rs_time_ns
    # exposure_time_ns
    # fov
    # virtual fov

    # Step 0: load the resulting poses.
    result_poses = LoadStabResult(result_poses_name)

    # Step 1: load the log data.
    # Process frame metadata.
    frame_data = LoadFrameData(frame_log_name)

    # Process gyro data.
    quats_data = LoadGyroData(gyro_log_name) 

    # Load OIS data.
    ois_data = LoadOISData(ois_log_name) 
    
    # Step 2: obtain per-frame per-scaneline projection matrices.
    num_frames = min(frame_data.shape[0], list(result_poses.values())[0].shape[0])
    grid = []
    for i in range(num_frames):
        metadata = GetMetadata(frame_data, i, result_poses)
        real_projections = GetProjections(static_options, metadata, quats_data, ois_data)
        virtual_projection = GetVirtualProjection(static_options, result_poses, metadata, i)
        grid.append(GetForwardGrid(static_options, real_projections, virtual_projection))
    grid = np.array(grid)
    # Now we confirm the grids are similar to the reference results.
    zoom_ratio = 1 / (1 - 2 * static_options["cropping_ratio"])
    curr_grid = CenterZoom(grid, zoom_ratio)
    
    curr_grid = np.transpose(curr_grid,(0,3,2,1))
    ref_grid = np.reshape(result_poses['warping grid'],(-1,12,12,4))

    # print(np.sum(np.abs(ref_grid - curr_grid)))
    # print(np.sum(np.abs(ref_grid) + np.abs(curr_grid)))

    # Step 3: generate frame to frame flow.
    for i in range(179,num_frames):
        real_projections_prev = GetProjections(static_options, GetMetadata(frame_data, i-1, result_poses), quats_data, ois_data)
        real_projections_curr = GetProjections(static_options, GetMetadata(frame_data, i, result_poses), quats_data, ois_data)
        forward_flow = GetWarpingFlow(
            real_projections_prev,real_projections_curr, len(real_projections_prev),len(real_projections_curr), 
            static_options["width"],static_options["height"])
        backward_flow = GetWarpingFlow(
            real_projections_curr,real_projections_prev, len(real_projections_curr),len(real_projections_prev), 
            static_options["width"],static_options["height"])
        print(forward_flow.shape)
        print(backward_flow)
        return
        plt.subplot(1,2,1)
        VisualizeFlow(forward_flow)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('forward flow frame ' + str(i))
        plt.subplot(122)
        VisualizeFlow(backward_flow)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('backward flow frame ' + str(i))
        plt.savefig("./python.jpg")
        return

def VisualizeFlow(flow):
    flow = np.reshape(flow, (-1), order='F') # Note reshape is different with matlab
    plt.plot(flow[0::4],flow[1::4],'r.')
    plt.plot(flow[2::4],flow[3::4],'g.')

if __name__ == "__main__":
    drive = "/mnt/disks/dataset/Google/test/"
    gyro_name = drive + 's2_outdoor_runing_forward_VID_20200304_144434/gyro_log_144425_189635.txt'
    ois_name = drive + 's2_outdoor_runing_forward_VID_20200304_144434/ois_log_144425_189635.txt'
    frame_name = drive + 's2_outdoor_runing_forward_VID_20200304_144434/frame_timestamps_144425_189635.txt'
    result = '/home/zhmeishi_google_com/dvs/data/testdata/results_updated/s2_outdoor_runing_forward_VID_20200304_144434_stab_mesh.txt'
    # GyroPlotDemo(gyro_name, ois_name, frame_name)
    EndToEndDemo(gyro_name, ois_name, frame_name, result)