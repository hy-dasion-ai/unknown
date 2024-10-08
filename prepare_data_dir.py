import os
import shutil

raw_data_dir = os.path.join(os.getcwd(), "data")


def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


from pydub import AudioSegment


def main():
    for file in os.listdir(raw_data_dir):
        name, ext = os.path.splitext(file)
        file_path = os.path.join(raw_data_dir, file)
        if ext == ".mp3":
            mp3_sound = AudioSegment.from_mp3(file_path)
            # rename them using the old name + ".wav"
            mp3_sound.export(os.path.join(raw_data_dir, "{0}.wav".format(name)), format="wav")

        if ext == ".wav":
            # create folder with file name
            folder = file.split(".")[0].strip()
            folder_path = os.path.join(raw_data_dir, folder)
            check_dir(folder_path)
            # create subfolder "speaker_clips"
            sub_folder_path = os.path.join(folder_path, "speaker_clips")
            check_dir(sub_folder_path)
            # create name replacement
            shutil.copyfile(file_path, os.path.join(sub_folder_path, "speaker_0.wav"))
            # delete original content
            os.remove(file_path)


if __name__ == '__main__':
    main()
    