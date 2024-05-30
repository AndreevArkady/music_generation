# Diploma: Generation of music accompaniment for videos based on the Transfomer architecture.

[Заметки после диплома Маши + работы Данила](https://snow-freon-ed9.notion.site/project-music-d5961ffef5374db99b07dcd236207301)

В этом репозитории нет запуска Video2Music

в video_conditioning лежит копия всего остального репозитория music-transformer :/

Что умеем? В ноутбуках:

#### my_version_of_auto_video_dataset.ipynb:
 content/, madmom/, tmp/, video_utils/, __base..., kvm..., script_mp3...

- .mp4 -> .mp3 -  moviepy.editor.VideoFileClip
- .mp3 -> instruments.wav & vocals.wav - vocal-remover
- .wav -> .mid - basic-pitch
---
- .mp4 -> key events timings list - Microsoft/computervision-recipes
- .mp4 -> key events timings list - joelibaceta/video-keyframe-detector
- 
- .wav -> genre - cetinsamet/music-genre-classification
- .mid -> velocity
- .wav / .mid -> tempo

- ? metrics ?

#### exec.ipynb:
 gen_script.py, model, code3, manual_generated, e_piano_small_kek

Запуск инференса music-transformer with nnotes



## Materials
Все должно быть в ноутбуках

video_data.zip - https://drive.google.com/file/d/1mbT2dq__3EbCHnfVtrP-JlJ0kEAWHTmt/view