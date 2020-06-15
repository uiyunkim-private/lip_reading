from google_drive_downloader import GoogleDriveDownloader as gdd
import os


def Dataset_downloader(name='gen1'):
    datasets = {'gen1':'1qeYPML6472pszI1auEOyboaXS5Rrlora'}

    gdd.download_file_from_google_drive(file_id=datasets['gen1'],
                                        dest_path='./data/dataset.zip',
                                        unzip=True)
    os.remove('./data/dataset.zip')