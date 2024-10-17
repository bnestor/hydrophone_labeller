"""
cli.py
"""


import hydra
from omegaconf import DictConfig, OmegaConf, listconfig
import os
from copy import deepcopy

from hydrophone_labeller.labeller import label_data



LOCAL_PATH = os.path.abspath(__file__)
print(LOCAL_PATH)


@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    """
    A command line tool that allows users to label hydrophone data.

    Args:
        classes (list): **required**. A list of classes that the user can label the data with.
        audio_files (str or list): **required**. The directory where the audio files are stored.
        save_dir (str): **required**. The directory where the labeled data will be saved.
        instructions (str): **required**. Instructions for the user.
        default_classes (list, optional): Default is ["multiple classes", "uncertain", "noise"]. 
            The default classes that the user can select from.
        objective_type (str, optional): Default is "multi-class". 
            The type of machine learning task. Options include "multi-class".
        sample_weights (str or None, optional): Default is None. 
            Path to a CSV containing columns "filename" and "sample weight".
        share (bool, optional): Default is False. 
            Whether to create a public sharable link for Gradio.
        deploy (bool, optional): Default is False. 
            Whether to deploy the labeller as a Gradio app.
        title (str, optional): Default is "hydrophone_labeller". 
            The title of the Gradio app.
    """


    
    if cfg.deploy:
        import huggingface_hub
        import shutil
        import json
        
       

        assert os.path.abspath(os.getcwd())==os.path.abspath(cfg.save_dir), print(f'In order to deploy this module, you must deploy it from the cfg.save_dir directory. The current directory is {os.getcwd()}, and the save_dir is {cfg.save_dir}')

        # assert that the directory is empty
        assert len(os.listdir(cfg.save_dir))==0, print('The directory must be empty to deploy the module')


        # create a dataset repository to upload the labelling data to.
        raise NotImplementedError('This feature is not yet implemented')



        # we need to save the hydra args to a yaml file in this directory
        print(os.path.dirname(__file__))
        print(os.getcwd())
        print(os.path.abspath('./outputs/deploy_folder'))
        deploy_cfg = deepcopy(cfg)
        del deploy_cfg.deploy
        deploy_cfg.save_dir = 'label_outputs/' # a huggingface spaces friendly location to save files
        deploy_cfg.audio_files = 'files_to_label'# a huggingface spaces friendly location to sync audio files to
        deploy_cfg.share = False

        deploy_cfg = OmegaConf.to_object(deploy_cfg)

        

        # the file to deploy is labeller.py
        # app_file = os.path.join(os.path.dirname(__file__), 'labeller.py')

        
        # it is easiest to make a temporary directory with the repo to upload to huggingface spaces.
        # os.makedirs(os.path.join(cfg['save_dir'], 'deploy_folder'), exist_ok=True)
        os.makedirs(os.path.join(cfg['save_dir'], 'deploy_folder','label_outputs'), exist_ok=True)
        shutil.copy(os.path.join(os.path.dirname(__file__), 'labeller.py'), os.path.join(cfg['save_dir'], 'deploy_folder','app.py'))
        os.chdir(os.path.join(cfg['save_dir'], 'deploy_folder'))

        # make a readme file with the cfg.title and cfg.instructions
        with open('README.md', 'w') as f:
            f.write(f"# {cfg.title}\n\n{cfg.instructions}")

        from gradio.cli.commands import deploy_space # this must come after the directory change to get the appropriate path

        # print(deploy_cfg['title'])
        deploy_cfg['title']=deploy_space.format_title(deploy_cfg['title'])
        # print(deploy_cfg['title'])



        # write deploy_cfg to a json file
        with open('deploy_cfg.json', 'w') as f:
            json.dump(deploy_cfg, f)



        # create a new repo
        deploy_space.deploy(title=cfg.title, app_file=os.path.join(os.getcwd(),'app.py'))

        


                        

        # upload config as environment variables for hf_api
        api = huggingface_hub.HfApi()
        whoami = api.whoami()
        for k, v in deploy_cfg.items():
            if isinstance(k, str):
                if k in ['classes', 'audio_files', 'save_dir', 'instructions', 'default_classes', 'objective_type', 'sample_weights']:
                    print(k.upper(), v)
                    api.delete_space_variable(repo_id=f"{whoami['name']}/{deploy_cfg['title']}",
                                           key=k.upper(), value=v)


    else:
        label_data(
            classes=OmegaConf.to_object(cfg.classes),
            audio_files=cfg.audio_files if isinstance(cfg.audio_files, str) else OmegaConf.to_object(cfg.audio_files), 
            save_dir=cfg.save_dir,
            instructions = cfg.instructions,
            default_classes = OmegaConf.to_object(cfg.default_classes),
            objective_type = cfg.objective_type,
            sample_weights = cfg.sample_weights,
            share = cfg.share,
        )




@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def compile_json(cfg: DictConfig):
    """
    For example:
    hydrophone-labeller-compile-labels --save_dir=outputs/label_outputs

    Args:
        save_dir (str): **required**. The directory where the labeled data will be saved.

    """
    import json
    import glob
    import polars as pl
    from collections import defaultdict



    all_json_files = glob.glob(os.path.join(cfg.save_dir, '*.json'))

    #'filename',  'user_hash',  'date_labelled', 'label'

    

    outputs = defaultdict(list)
    for filename in all_json_files:
        with open(filename, 'r') as f:
            data = json.load(f)
        if not(isinstance(data, dict)):
            continue
        if not(all([k in data.keys() for k in ['filename', 'user_hash', 'date_labelled', 'label']])):
            continue

        for k,v in data.items():
            outputs[k].append(v)

    new_df = pl.DataFrame(outputs)

    if not(os.path.exists(os.path.join(cfg.save_dir, 'labelled_data.csv'))):
        # create a csv file with the headers
        df = pl.read_csv(os.path.join(cfg.save_dir, 'labelled_data.csv'))

        new_df = pl.concat((df, new_df))

        new_df = new_df.drop_duplicates()


    if len(new_df)==0:
        print('No new labels to compile')
        return
    
    new_df.write_csv(os.path.join(cfg.save_dir, 'labelled_data.csv'))

    print(f"There are {len(new_df)} labels covering {len(new_df.select('filename').unique())} files")


@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def prepare_data(cfg: DictConfig):
    """
    Example:
    ```
    hydrophone-labeller-prepare-data audio_files=<path/to/audio/files/*.flac> processed_outputs=</path/to/processed/flac>
    ```

    Args:
        audio_files (str or list): **required**. The directory where the audio files are stored.
        start_segments (str, optional): A csv with the filename and the start time for each clip. If this is not provided, the first 15 seconds are used
        processed_outputs (str): **required**. The directory where the processed audio files will be saved
    """
    from hydrophone_labeller.dataset_preparation import prepare_data



    prepare_data(
        audio_files = cfg.audio_files,
        start_segments = cfg.start_segments,
        processed_outputs = cfg.save_dir,
    )






if __name__ == "__main__":
    main()

