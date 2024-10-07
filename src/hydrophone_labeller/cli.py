"""
cli.py
"""


import hydra
from omegaconf import DictConfig
import os


from hydrophone_labeller.labeller import label_data


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):



    
    
    label_data(
        classes=cfg.classes,
        audio_files=cfg.audio_files, 
        save_dir=cfg.save_dir,
        default_classes = cfg.default_classes,
        objective_type = cfg.objective_type,
        sample_weights = cfg.sample_weights,
        share = cfg.share,
        deploy = cfg.deploy,
    )



if __name__ == "__main__":
    main()

