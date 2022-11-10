import GPUtil
from nemo.utils import model_utils
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
import os
from glob import glob
import torch
from pyctcdecode import build_ctcdecoder

import src.media_convertor
from src import utilities, log_setup
from src.lib.audio_normalization import AudioNormalization
from src.monitoring import monitor
from src.srt.timestamp_generator import extract_time_stamps
from src.utilities import get_env_var

LOGGER = log_setup.get_logger(__name__)

def get_cuda_device():
    gpu = get_env_var('gpu', 'false')
    if gpu == 'true' or gpu == 'True':
        gpu = True
    else:
        gpu = False

    gpu_present = torch.cuda.is_available()
    LOGGER.info(f'User has provided gpu as {gpu} gpu_present {gpu_present}')

    gpu = gpu & gpu_present

    if gpu:
        LOGGER.info('### GPU Utilization ###')
        GPUtil.showUtilization()
        available_devices = get_env_var('CUDA_VISIBLE_DEVICES', '0').split(',')
        all_gpu = [g.id for g in GPUtil.getGPUs()]
        req_gpu = [int(a) for a in available_devices]
        excluded_gpus = list(set(all_gpu) - set(req_gpu[1:]))
        LOGGER.info(f'available GPUs {available_devices}, all GPUs {all_gpu}, excluded GPUs {excluded_gpus}')
        # Example:  env gives CUDA_VISIBLE_DEVICES = 2,5
        # mapped GPU on pod will be {0: 2, 1:5}
        # all_gpu = 0,1,2,3,4,5,6,7 then we will exclude 0,2,5
        # excluded_gpus will be 0,1,3,4,6,7
        # we are skipping 0th GPUs as punctuation model be default goes on 0th GPU
        selected_gpus = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.8, maxMemory=0.75,
                                            excludeID=excluded_gpus)
        LOGGER.info(f'Selected GPUs: {selected_gpus} requested GPUs {req_gpu}')
        if len(selected_gpus) > 0 and selected_gpus[0] in req_gpu:
            selected_gpu_index = req_gpu.index(selected_gpus[0])
        else:
            selected_gpu_index = None

        selected_gpu = torch.device("cuda", selected_gpu_index)
        LOGGER.info(f'selected gpu index: {selected_gpu_index} selecting device: {selected_gpu}')
        return selected_gpu
    else:
        return None


SELECTED_DEVICE = get_cuda_device()

@monitor
def get_results(wav_path, dict_path, generator, use_cuda=False, w2v_path=None, model=None, half=None):

    dir_name = src.media_convertor.media_conversion(wav_path, duration_limit=15)
    audio_file = dir_name / 'clipped_audio.wav'

    sample = utils.move_to_cuda(sample, SELECTED_DEVICE) if use_cuda else sample
    logits = model.transcribe([audio_file], logprobs=True)[0]

    text = generator.deode(logits)
    LOGGER.debug(f"deleting sample...")
    del sample
    if use_cuda:
        LOGGER.debug(f"clearing cuda cache...")
        torch.cuda.empty_cache()
    LOGGER.debug(f"infer completed {text}")

    return text

def load_model_and_generator(model_item, cuda, decoder="viterbi"):
    model_path = model_item.get_model_path()
    lexicon_path = model_item.get_lexicon_path()
    lm_path = model_item.get_language_model_path()

    model_cfg = ASRModel.restore_from(restore_path=model, return_config=True)
    classpath = model_cfg.target
    imported_class = model_utils.import_class_by_path(classpath)

    if cuda:

        with torch.cuda.device(SELECTED_DEVICE):
            LOGGER.info(f'using current device: {torch.cuda.current_device()}')
            model = imported_class.restore_from(restore_path=model_path, map_location=SELECTED_DEVICE)

        ln_code = model_item.get_language_code()
        LOGGER.info(f"{ln_code} Model initialized with GPU successfully")

    else:
        model = imported_class.restore_from(restore_path=model_path)


    if decoder == "viterbi":
        generator = build_ctcdecoder(model.decoder.vocabulary)

    else:
        with open(lexicon_path, encoding='utf-8') as f:
            unigram_list = [t for t in f.read().strip().split('\n')]

        generator = build_ctcdecoder(model.decoder.vocabulary, lm_path, unigram_list)

    LOGGER.info(f'Loading model from {model_path} cuda {cuda}')

    return model, generator
