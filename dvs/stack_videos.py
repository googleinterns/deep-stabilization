import subprocess
import os, sys


video0_path = sys.argv[1]
video0_label = sys.argv[2]
video1_path = sys.argv[3]
video1_label = sys.argv[4]
output_video_path = sys.argv[5]


# Generate video mosaic.
cmd = ('ffmpeg -y -loglevel error -i {v0} -i {v1} '
       '-filter_complex '
       '\"[1:v][0:v]scale2ref=oh*mdar:ih[1v][0v];'
       '[0v]drawtext=text=\'{t0}\':fontcolor=white:fontsize=40:'
       'x=10:y=(h-text_h-10):box=1:boxcolor=black@0.5:boxborderw=5[0vt];'
       '[1v]drawtext=text=\'{t1}\':fontcolor=white:fontsize=40:'
       'x=10:y=(h-text_h-10):box=1:boxcolor=black@0.5:boxborderw=5[1vt];'
       '[0vt][1vt]hstack=2\"'
       ' -an {vo}').format(
            v0=video0_path,
            v1=video1_path,
            t0=video0_label,
            t1=video1_label,
            vo=output_video_path)
print(cmd)
subprocess.call(cmd, shell=True)