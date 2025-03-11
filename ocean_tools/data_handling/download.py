# ocean_tools/data_handling/download.py

import os
import time
import shutil
import webbrowser
import requests
from pathlib import Path

def wait_for_and_move_file(file_name: str, destination_path: str, timeout: int = 60, verbose: bool = False) -> None:
    """
    Waits for a specified file to appear in the user's Downloads folder and then moves it to a destination.

    This function continuously checks the Downloads directory for the presence of the specified file.
    Once the file is found (or the timeout is exceeded), it is moved to the destination path.

    Parameters
    ----------
    file_name : str
        The name of the file to wait for.
    destination_path : str
        The full path (including file name) where the file should be moved.
    timeout : int, optional
        Maximum time (in seconds) to wait for the file. Default is 60.
    verbose : bool, optional
        If True, prints status messages during the waiting process. Default is False.

    Raises
    ------
    FileNotFoundError
        If the file is not found in the Downloads folder within the specified timeout.

    Returns
    -------
    None
    """
    downloads_dir = str(Path.home() / "Downloads")
    file_path = os.path.join(downloads_dir, file_name)
    destination_file = Path(destination_path)

    start_time = time.time()

    while not os.path.exists(file_path):
        elapsed_time = int(time.time() - start_time)
        if elapsed_time > 0 and int(elapsed_time) % 5 == 0 and verbose:
            print(f"Waiting {int(elapsed_time)} seconds for {file_name}...")
        time.sleep(1)
        if elapsed_time > timeout:
            raise FileNotFoundError(
                f"File '{file_name}' not found in {downloads_dir} after {timeout} seconds."
            )

    destination_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(file_path, destination_path)
    if verbose:
        print(f"File moved to {destination_path}")


def download_image_from_url(image_url: str, destination_path: str, verbose: bool = False) -> None:
    """
    Downloads an image file (PNG) from a given URL and saves it to the specified destination.

    Parameters
    ----------
    image_url : str
        The URL of the image to download.
    destination_path : str
        The file path where the downloaded image will be saved.
    verbose : bool, optional
        If True, prints a success message upon completion. Default is False.

    Raises
    ------
    requests.exceptions.RequestException
        If the HTTP request for downloading fails.

    Returns
    -------
    None
    """
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        with open(destination_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        if verbose:
            print(f"Image downloaded successfully to {destination_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image: {e}")
        raise


def download_file_using_browser_and_move(in_url: str, destination_path: str, timeout: int = 60, verbose: bool = False) -> None:
    """
    Downloads a file by opening its URL in the default browser (or downloads directly if the file is a PNG)
    and moves it to the specified destination path.

    For PNG files, the image is downloaded directly. For other file types, the URL is opened in the browser
    and the function waits for the file to appear in the Downloads folder before moving it.

    Parameters
    ----------
    in_url : str
        The URL of the file to download.
    destination_path : str
        The destination directory where the file should be moved.
    timeout : int, optional
        Maximum time (in seconds) to wait for the file to appear in the Downloads folder. Default is 60.
    verbose : bool, optional
        If True, prints status messages. Default is False.

    Returns
    -------
    None
    """
    file_name = in_url.split("/")[-1]
    output_path = os.path.join(destination_path, file_name)

    if file_name.lower().endswith(".png"):
        download_image_from_url(in_url, output_path, verbose)
    else:
        if verbose:
            print(f"Opening the URL in your default browser: {in_url}")
        webbrowser.open(in_url, new=2, autoraise=False)
        wait_for_and_move_file(file_name, output_path, timeout, verbose)


def bulk_download_files(file_urls: str, destination_path: str, max_files: int = 0, file_timeout: int = 60, verbose: bool = False) -> None:
    """
    Downloads multiple files from a list of URLs provided as a string and saves them to the specified destination.

    The function reads a newline-separated list of URLs, filters out files that already exist in the destination,
    and downloads each file using the browser (or directly if it is a PNG). It tracks and prints progress.

    Parameters
    ----------
    file_urls : str
        A string containing URLs (one per line) of the files to download.
    destination_path : str
        The directory where the downloaded files will be stored.
    max_files : int, optional
        The maximum number of files to download. If 0, all pending files are downloaded. Default is 0.
    file_timeout : int, optional
        Maximum time (in seconds) to wait for each file to appear in the Downloads folder. Default is 60.
    verbose : bool, optional
        If True, prints progress and status messages. Default is False.

    Returns
    -------
    None
    """
    url_list = list(filter(None, map(str.strip, file_urls.split("\n"))))
    n_total_files = len(url_list)

    # No se descargan archivos ya presentes en la carpeta de destino
    destination_files = os.listdir(destination_path)
    url_list = [url for url in url_list if url.split("/")[-1] not in destination_files]
    
    n_pending_files = len(url_list)
    n_skippable_files = n_total_files - n_pending_files

    if max_files > 0 and max_files < n_pending_files:
        url_list = url_list[:max_files]

    n_download_files = len(url_list)
    
    print(f"Total: {n_total_files} | Existing: {n_skippable_files} | Pending: {n_pending_files} | Downloading: {n_download_files}")
    start_time = time.time()

    effectively_downloaded = 0
    for i, url in enumerate(url_list, start=1):
        try:
            download_file_using_browser_and_move(url, destination_path, file_timeout, verbose)
            effectively_downloaded += 1
        except FileNotFoundError as e:
            print(f"Error downloading file: {e}")

        if verbose or i % 10 == 0 or i == n_download_files:
            print(f"Processed: {i} | Downloaded {effectively_downloaded} | Run time: {int(time.time() - start_time)}s")

    print(f"{effectively_downloaded} files downloaded successfully to {destination_path}.")

    if effectively_downloaded < n_download_files:
        print(f"{n_download_files - effectively_downloaded} files failed to download.")