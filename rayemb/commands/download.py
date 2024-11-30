import click
import os

import requests
from tqdm import tqdm

from rayemb.utils import setup_logger
from rayemb.constants import DEEPFLUORO_DATA_URL, RAYEMB_CTPELVIC1K_CHECKPOINT_URL

logger = setup_logger(__name__)

@click.group()
def download():
    pass

@download.command()
@click.option('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save the checkpoints')
@click.option('--model', type=click.Choice(["rayemb"]), help='Which model checkpoint to download', default='rayemb')
@click.option('--dataset', type=click.Choice(['deepfluoro', 'ctpelvic1k']), help='Dataset to download', default='ctpelvic1k')
def checkpoint(checkpoint_dir, model, dataset):
    logger.info(f'Downloading {model} checkpoint for {dataset}...')
    os.makedirs(checkpoint_dir, exist_ok=True)

    if model == 'rayemb':
        if dataset == 'ctpelvic1k':
            response = requests.get(RAYEMB_CTPELVIC1K_CHECKPOINT_URL, stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                with open(os.path.join(checkpoint_dir, 'rayemb-ctpelvic1k.ckpt'), 'wb') as f, tqdm(
                    desc='Downloading',
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                        bar.update(len(chunk))
                logger.info(f'Download complete. File saved to {os.path.join(checkpoint_dir, "rayemb-ctpelvic1k.ckpt")}')
            else:
                logger.error(f'Failed to download file. Status code: {response.status_code}')
        else:
            raise NotImplementedError(f'{model} checkpoint for {dataset} not yet available')
    else:
        raise NotImplementedError(f'{model} checkpoint not yet available')

@download.command()
@click.option('--data_dir', type=str, default='./data', help='Directory to save the data')
@click.option('--dataset', is_flag=False, help='Generated dataset or the original DeepFluoro data')
def deepfluoro(data_dir, dataset):
    if dataset:
        raise NotImplementedError('Generated dataset not yet available')
    else:
        logger.info('Downloading the original DeepFluoro data...')
        os.makedirs(data_dir, exist_ok=True)

        response = requests.get(DEEPFLUORO_DATA_URL, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            with open(os.path.join(data_dir, 'ipcai_2020_full_res_data.h5'), 'wb') as f, tqdm(
                desc='Downloading',
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
                    bar.update(len(chunk))
            logger.info(f'Download complete. File saved to {os.path.join(data_dir, "ipcai_2020_full_res_data.h5")}')
        else:
            logger.error(f'Failed to download file. Status code: {response.status_code}')