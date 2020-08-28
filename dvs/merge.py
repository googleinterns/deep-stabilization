import os
import sys

# ffmpeg path
ffmpeg_path = 'ffmpeg'

# print(sys.argv)

            
# if len(sys.argv) == 5:
#   f0 = sys.argv[1]
#   f1 = sys.argv[2]
#   f2 = sys.argv[3]
#   fo = sys.argv[4]
  
#   command = ffmpeg_path + ' -i {f0} -i {f1} -i {f2} -y -filter_complex ' \
#             '"[0:v]pad=iw*3:ih[a]; [a][1:v]overlay=w[x]; [x][2:v]overlay=2*w:0" -b:v 60000k {fo}'\
#             .format(f0=f0, f1=f1, f2=f2, fo=fo)
    
# if len(sys.argv) == 6:
#   f0 = sys.argv[1]
#   f1 = sys.argv[2]
#   f2 = sys.argv[3]
#   f3 = sys.argv[4]
#   fo = sys.argv[5]
  
#   command = ffmpeg_path + ' -i {f0} -i {f1} -i {f2} -i {f3} -y -filter_complex ' \
#             '"[0:v]pad=2*iw:2*ih[a]; [a][1:v]overlay=w[x]; [x][2:v]overlay=0:h[y]; [y][3:v]overlay=w:h" -b:v 120000k {fo}'\
#             .format(f0=f0, f1=f1, f2=f2, f3=f3, fo=fo)
    

# path1 = "/home/zhmeishi_google_com/presentation/data_google/"
# videos = sorted(os.listdir(path1))
# for i in range(len(videos)):
#   video_name = videos[i][:-4]
#   f0 = "/home/zhmeishi_google_com/presentation/data_google/" + video_name + ".mp4"
#   f1 = "/home/zhmeishi_google_com/presentation/CVPR2020/" + video_name + "_result.mp4"
#   f2 = "/home/zhmeishi_google_com/presentation/EIS2020/" + video_name + "_stab.mp4"
#   f3 = "/home/zhmeishi_google_com/presentation/opt_base3_continue_75_20_5/" + video_name + "_stab.mp4"
#   fo = "/home/zhmeishi_google_com/presentation/combine/" + video_name + "_combine.mp4"

#   command = ffmpeg_path + ' -i {f0} -i {f1} -i {f2} -i {f3} -y -filter_complex ' \
#             '"[0:v]pad=2*iw:2*ih[a]; [a][1:v]overlay=w[x]; [x][2:v]overlay=0:h[y]; [y][3:v]overlay=w:h" -b:v 120000k {fo}'\
#             .format(f0=f0, f1=f1, f2=f2, f3=f3, fo=fo)
#   os.system(command)

video_name = "indoor-walk_hallway_VID_20191216_193554"
# f0 = "/home/zhmeishi_google_com/presentation/indoor_data/" + video_name + ".mp4"
f0 = "/home/zhmeishi_google_com/presentation/indoor_EIS/" + video_name + "_stab.mp4"
f1 = "/home/zhmeishi_google_com/presentation/indoor_opt/" + video_name + "_stab.mp4"
fo = "/home/zhmeishi_google_com/presentation/indoor_combine/" + video_name + "_combine.mp4"
# f0 = "/home/zhmeishi_google_com/presentation/indoor_data/dR_run_fps60.mp4"
# f1 = "/home/zhmeishi_google_com/presentation/opt_base3_continue_75_20_5/s2_outdoor_runing_forward_VID_20200304_144434_stab_fps60.mp4"
# fo = "/home/zhmeishi_google_com/presentation/select/image_dR_run.mp4"

# combine input and output videos into a single side-by-side video
command = ffmpeg_path + ' -i {f0} -i {f1} -y -filter_complex ' \
          '"[0:v:0]pad=2*iw:ih[bg]; [bg][1:v:0]overlay=w" -b:v 60000k {fo}' \
          .format(f0=f0, f1=f1, fo=fo)
# command = ffmpeg_path + ' -i {f0} -i {f1} -i {f2} -y -filter_complex ' \
#             '"[0:v]pad=iw*3:ih[a]; [a][1:v]overlay=w[x]; [x][2:v]overlay=2*w:0" -b:v 60000k {fo}'\
#             .format(f0=f0, f1=f1, f2=f2, fo=fo)

os.system(command)
