"""
labeller.py

A gradio app that allows users to label hydrophone data.
"""

import gradio as gr
import pandas as pd
import os

import glob

import torchaudio
import hashlib
import json
from datetime import datetime



# Utility function to create a hash from the user ID
def generate_user_hash(user_id):
    return hashlib.sha256(user_id.encode()).hexdigest()[:8]

# Function to save the label data
def save_label_data(save_dir, filename, user_hash, label, generator):
    date_labelled = datetime.now().isoformat()
    label_data = {
        'filename': filename,
        'user_hash': user_hash,
        'date_labelled': date_labelled,
        'label': label
    }
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{os.path.basename(filename).replace('.mp3','').replace('.flac','').replace('.wav','')}_{user_hash}.json")
    if not(os.path.exists(file_path)):
        with open(file_path, 'w+') as f:
            json.dump(label_data, f)


    # move on to the next file

    audio_file = next(generator)
    spectrogram = create_spectrogram(audio_file)


    return audio_file, spectrogram, os.path.basename(audio_file), os.path.basename(audio_file)

# Function to generate a spectrogram
def create_spectrogram(filename):
    assert os.path.exists(filename.replace('.wav', '.png').replace('.mp3', '.png').replace('.flac', '.png'))
    return filename.replace('.wav', '.png').replace('.mp3', '.png').replace('.flac', '.png')


# # Initial user ID prompt
# def request_user_id(user_id, classes, audio_file, save_dir):
#     if not user_id:
#         return "Please enter a valid ID."

#     user_hash = generate_user_hash(user_id)
#     spectrogram_image = create_spectrogram(audio_file)
#     return gr.update(visible=False), gr.update(visible=True), user_hash, spectrogram_image

# # Handle label submission
# def submit_label(user_hash, audio_file, save_dir, label):
#     date_labelled = datetime.now().isoformat()
#     save_label_data(save_dir, audio_file, user_hash, label, date_labelled)
#     return f"Label '{label}' saved successfully."

def create_labeling_interface(classes):
    if len(classes) <= 6:
        return gr.Group([gr.Button(label) for label in classes])
    else:
        return gr.Dropdown(classes, label="Choose class")
    

def image_audio_generator(audio_files):
    for audio_file in audio_files:
        yield audio_file





# Function to start the labeling process after ID input
def start_labelling(user_id, classes, audio_files, save_dir):
    print("in start labelling")
    user_hash = generate_user_hash(user_id)
    print('User hash:', user_hash)
    # set user hash to state
    # audio_file = audio_files[0]  # Assuming one file for now, or you can iterate
    user_audio_files = [audio_file for audio_file in audio_files if not os.path.exists(os.path.join(save_dir, f"{os.path.basename(audio_file).replace('.mp3','').replace('.flac','').replace('.wav','')}_{user_hash}.json"))]

    audio_file = user_audio_files[0]

    audio_file_generator = iter(user_audio_files[1:])
    
    spectrogram = create_spectrogram(audio_file)
    return gr.update(visible=False), gr.update(visible=True), spectrogram, audio_file, user_hash, os.path.basename(audio_file), os.path.basename(audio_file), audio_file_generator


# def handle_button_click(user_hash, file_name, label, save_dir):
#     return submit_label(user_hash, file_name, label, save_dir)

# Build the Gradio interface
def label_data(classes, audio_files, save_dir, instructions, default_classes=["multiple classes","uncertain","noise"], objective_type="multiclass", sample_weights=None,  share=False, deploy=False):
    """
    behaviour: upon entering a unique name,
    search through all instances of *_userhash.json and remove these from the list
    create a generator for a list of the audio files

    Inputs:
        classes: list of str, classes to be labelled
        audio_files: str, list, specifying path to audio files
        save_dir: str, path to directory to save labelled data
        include_columns: list of str, columns to include in the classes by default
        objective_type: str, type of objective (e.g. multiclass, multilabel)
        sample_weights: a file with the basename of each audio_file, and a corresponding weight for the probability of displaying it to the user. samples not contained in this directory are automatically labelled as the minimum probability
        share: bool, whether to generate a link to share the app from the local machine
        deploy: bool, whether to deploy the app to huggingface spaces
    Returns:
        None
    """
    assert objective_type=="multiclass", print('Only multiclass objective supported for now')

    # add directory for gradio to expose
    

    classes = sorted(list(set(classes)-set(default_classes)))+default_classes


    # collect the data
    if isinstance(audio_files, str):
        if '*' in audio_files:
            filenames = glob.glob(audio_files, recursive=True)
        else:
            assert os.path.isfile(audio_files), print(audio_files, 'does not exist')
            filenames = [audio_files]
    elif isinstance(audio_files, list): # or isinstance(audio_files, listconfig.ListConfig):
        filenames=[]
        for f in audio_files:
            if '*' in f:
                print('globbing', f)
                filenames.extend(glob.glob(f, recursive=True))
            else:
                assert os.path.isfile(f), print(f, 'does not exist')
                filenames.append(f)

    audio_files = filenames

    assert len(audio_files) > 0, print('No audio files found')

    gr.set_static_paths(paths=audio_files+[f.replace(".mp3",".png") for f in audio_files])
    

    # spec_audio_gen = image_audio_generator(audio_files)

    # def start():
    #     filename, spectrogram, image = next(spec_audio_gen)
    #     return filename, spectrogram, image


    # if sample_weights is not None then sort the audio files by the sample weights
    if sample_weights is not None:
        import csv
        with open(sample_weights, 'r') as f:
            reader = csv.reader(f)
            sample_weights = {row[0]:row[1] for row in reader}
        sample_weights = {k:float(v) for k,v in sample_weights.items() if v.isnumeric()}

        sample_weights = {os.path.basename(k):v for k,v in sample_weights.items()}
        # descending
        audio_files = sorted(audio_files, key=lambda x: sample_weights.get(os.path.basename(x), 0), reverse=True)
    
    
    
    with gr.Blocks() as block:
        # Initial screen for user ID
        with gr.Row(visible=True) as start_interface:
            intro_text = gr.Markdown(instructions)
            user_id_input = gr.Textbox(label="Enter User ID. This value will be hashed for privacy. This ensures that we can attribute the source of labels.", placeholder="User ID")
            start_button = gr.Button("Start Labeling")
        




        with gr.Row(visible=False) as labelling_interface:
            with gr.Column(scale=3):
                file_name = gr.Markdown(visible=True)
                image_block = gr.Image(visible=True)
                audio_block = gr.Audio(visible=True)

            with gr.Column(scale=1):
                # label_buttons = [gr.Button(label) for label in classes]
                # file_name = gr.Textbox(info="Filename", visible=True)
                if len(classes) <= 9:
                    label_buttons = [gr.Button(label) for label in classes]
                else:
                    label_dropdown = gr.Dropdown(classes, label="Choose class")

        user_hash = gr.State()
        current_audio_file = gr.State()
        user_audio_file_generator = gr.State(value=audio_files)

        # Print statements for debugging (should not access State directly)
        def print_debug_values(user_hash, file_name):
            print(f"User Hash: {user_hash}, File Name: {file_name}")
            return gr.update(visible=True)

        # Handle button clicks for labeling
        if len(classes) <= 9:
            for label_button in label_buttons:
                label_button.click(
                    fn=save_label_data,
                    inputs=[gr.State(save_dir), current_audio_file, user_hash, gr.State(value=label_button.value), user_audio_file_generator ],
                    # inputs=[user_hash, current_audio_file, gr.State(value=label_button.value), gr.State(save_dir)],
                    outputs=[audio_block, image_block, file_name, current_audio_file,]
                )
        else:
            label_dropdown.change(
                fn=save_label_data,
                # inputs=[user_hash, current_audio_file, label_dropdown, gr.State(save_dir)],
                inputs=[gr.State(save_dir), current_audio_file, user_hash, gr.State(value=label_dropdown.value), user_audio_file_generator ],
                outputs=[audio_block, image_block, file_name, current_audio_file]
            )


        start_button.click(fn=start_labelling, #start 
                           inputs = [user_id_input, gr.State(classes), gr.State(audio_files), gr.State(save_dir)],
                           outputs=[start_interface, labelling_interface, image_block, audio_block, user_hash, file_name, current_audio_file, user_audio_file_generator]
                           )
        
        # for label_button in iterator:
        #     label_button.click(fn=submit_label,
        #                        inputs = [user_hash, current_audio_file, label_button, save_dir],
        #                        outputs=[file_name], user_hash=gr.State(), audio_file=gr.State(), save_dir=gr.State(), label=label_button.value)

        

    # block.queue()
    block.launch(share=share, allowed_paths=[save_dir,])
    # app_interface = gr.Interface(
    #     fn=request_user_id,
    #     inputs=[user_id_input, gr.State(classes), gr.State(audio_files), gr.State(save_dir)],
    #     outputs=[user_id_input, spectrogram_image, gr.State(), gr.Image(label="Spectrogram")],
    # )
    
    # app_interface.launch( share=share)


if __name__=="__main__":
    # if it is deployed, we should have exported the args to yaml.
    # yaml args will be read here to feed to the model
    import json

    with open('deploy_config.json','r') as f:
        deploy_cfg = json.load(f)
    

    


    label_data(
        classes = deploy_cfg['classes'],
        audio_files = deploy_cfg['audio_files'], 
        save_dir = deploy_cfg['save_dir'],
        instructions = deploy_cfg['instructions'],
        default_classes = deploy_cfg['default_classes'],
        objective_type = deploy_cfg['objective_type'],
        sample_weights = deploy_cfg['sample_weights'],
    )

