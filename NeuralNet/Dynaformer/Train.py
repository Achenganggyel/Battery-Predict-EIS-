'''
from download_model_and_sample_data.py
@modifier: Alla
'''
import sys
sys.path.append('../../')
from dataLoader import DataLoader
from TrainAndPredict import Process
from Dynaformer import Dynaformer
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
import Config

# Since cannot get weights of the model through URL by Python, 
# thus directly downloaded it
class Train():
    # initialize data
    # TODO：感觉它数据格式有些复杂啊，先看别的了
    def __init__(self, X_path='', y_path='', calPath='', X_pre_path=''):
        thisLoader = DataLoader()

    # use config to set models
    def prepare(self, config=None):
        if config.model_name =="dynaformer":
            model = Dynaformer(final_out_dim=1,
                            lr=config.model_lr,
                                is_instance_norm=config.model_instance_norm, 
                                loss=config.model_loss, 
                                patience_lr_plateau=config.model_patience_lr_plateau)
            save_top_k = 1
            drop_final=True
        else:
            raise Exception("Cannot recognize the model")
    
    # aaa

def main(model_type):
    config = Config()
    train = Train()

if __name__ == "__main__":
    pass