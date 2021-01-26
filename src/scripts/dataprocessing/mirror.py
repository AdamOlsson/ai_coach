import os, ffmpeg

def mirror(path):
    video = [f for f in os.listdir(path) if ".mp4" in f]

    for v in video:
        name = v.split('.')[0]
        savepath = os.path.join(path, name + "_mirrored.mp4")
        stream = ffmpeg.input(os.path.join(path, v))
        stream = ffmpeg.hflip(stream)
        stream = ffmpeg.output(stream, savepath)
        ffmpeg.run(stream)



    


if __name__ == "__main__":
    mirror("/media/adam/G/datasets/weightlifting/telegram2/videos/data/power_snatch")