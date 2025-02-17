# ocean_tools/data_handling/download.py

import os
import time
import shutil
import webbrowser
import requests
from pathlib import Path

def wait_for_and_move_file(file_name: str, destination_path: str, timeout: int = 60, verbose: bool = False) -> None:
    """
    Espera a que aparezca un archivo en la carpeta de descargas 
    y lo mueve a la ruta especificada.
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
    Descarga un archivo de imagen (PNG) desde una URL y lo guarda en la ruta indicada.
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
    Abre una URL en el navegador por defecto para descargar el archivo 
    y luego lo mueve a la carpeta de destino. 
    Si el archivo es .png, se descarga directamente.
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
    Descarga múltiples archivos abriendo sus URLs en el navegador (o directamente si .png)
    y los mueve a la carpeta de destino.
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
