from pytube import YouTube

#Video URL that needs to be downloaded
link = "https://www.youtube.com/watch?v=zV1qLYukTH8"


yt = YouTube(link) 
title = yt.title

name_arr = title.lower().strip().split()
final_name = "_".join(name_arr[0:min(5, len(name_arr))]) + ".mp4"


try:
    yt.streams.filter(adaptive = True).first().download(output_path = "/data/videos/", filename = final_name)
    #yt.streams.filter(res = "1080p", progressive = True, file_extension = "mp4").first().download(output_path = "videos/", filename = "SugarPlumFairy.mp4")
except:
    print("Some Error!")
print('Task Completed!')

