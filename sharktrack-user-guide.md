# SharkTrack User Guide
*How to run SharkTrack on your BRUVS...*

## Contents

* <a href="#overview">Overview</a>
* <a href="#our-ask-to-sharktrack-user">Our Ask to SharkTrack user</a>
* <a href="#quick-tutorial">Quick Tutorial</a>
* <a href="#step-by-step">Detailed Guide</a>
* <a href="#how-fast-is-sharktrack-can-i-use-it-on-my-laptop">How fast is SharkTrack Can I use it on my laptop?</a>
* <a href="#model-types-mobile-vs-analyst">Model Types: Mobile vs Analyst</a>
* <a href="#can-i-trust-its-accuracy">Can I trust it's accuracy?</a>
* <a href="#next-steps">Next steps</a>

## Overview
This page provides a guide on running the SharkTrack ML model on your BRUVS videos to detect shark in them. 

> If you already have an output and want to comput MaxN, jump to [this guide](./annotation-pipelines.md)

<img src="./static/test-output/detections/11.jpg" width=400/>

*An example of SharkTrack output detection. Learn more [here](./annotation-pipelines.md#step-0-understand-the-output)*


## Our Ask to SharkTrack user
SharkTrack is free, and it makes us super-happy when people use it, so we put it out there as a downloadable model that is easy to use. That means we don't know who's using it unless you contact us, so please please [email us](mailto:fppvrn@gmail.com?subject=SharkTrackUser) and star this repo if you find it useful!

## Quick Tutorial
If you don't have experience with Python or you find any error with the following instructions, move to the [Step by Step](#step-by-step) guide.
1. Clone SharkTrack
    ```bash
    git clone https://github.com/filippovarini/sharktrack.git
    cd sharktrack
    ```
2. Setup Virtual Environment
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
    See [troubleshooting](#troubleshooting) for problems
3. Run the model
    ```bash
    python app.py
    ```
    This will run the model against the example video in `./input_videos`

## Step By Step
If you don't have experience with Python, the following steps might be the hardest to setup the model. If you think your organisation would benefit from near-instant BRUVS analysis, please consider this challenge as an investment in the future. Don't hesitate to post any problems you might encounter [here](https://github.com/filippovarini/sharktrack/issues).

### Pre-Requirements
To run sharktrack, you will need to have installed Python 3.9 or above. You can check that by opening your "Command Prompt" application on Windows or "Terminal" on Mac and running `python -V`. *Note: on Mac the older version might be saved as `python3 -V`*

If you don't have python or the version is outdated, please install it [here](https://www.python.org/downloads/). Make sure to check the box to add Python to the system path when installing it.

### Downloading the model
Follow the below instructions to download the model. Alternatively, if you are familiar with `git`, skip to [here](#quick-tutorial).

1. Download the model from [here](https://github.com/filippovarini/sharktrack/releases) by double-clicking on the latest "source code". Dowload the zipfile only!
2. Unzip the folder and extract `sharktrack`. We suggest moving the downloaded folder to the Desktop or any place where it is easier to find.
3. Unzip the downloaded folder and
    - If on **Windows**:
        - Open the extracted `sharktrack` folder
        - If you use anaconda:
            - Click on the [address bar](https://uis.georgetown.edu/wp-content/uploads/2019/05/win10-fileexplorer-addrbar.png)
            - Click "Copy address" in the address bar
            - Open the Anaconda Prompt
            - Run `cd {the address you just copied}`
        - If you use Python from the Command Prompt
            - Click on the [address bar](https://uis.georgetown.edu/wp-content/uploads/2019/05/win10-fileexplorer-addrbar.png)
            - Replace the text with `cmd` and hit Enter
        - This sholuld show you the Command/Anaconda Prompt application
    - If on **Mac**:
        - Right-click on the `sharktrack` folder
        - Click "New Terminal at Folder" (sometimes this is under Services)
        - This should show you the Terminal application
    

### Setup Environment
Now you have the downloaded the model and you have the Terminal application open. It is time to setup the software environment.
- **Windows**
    ```
    python -m venv venv
    venv\Scripts\activate.bat
    pip install -r requirements.txt
    ```
- **Mac**
    
    1. Firstly, copy-paste in Terminal the following command and hit enter:
        ```
        python -m venv venv
        ```
        If it fails, try copy-pasting this command instead `python3 -m venv venv`.
    2. Secondly, copy-paste in Terminal the following command and hit enter:
        ```
        source venv/bin/activate
        pip install -r requirements.txt
        ```
    
- **Anaconda**
    ```
    conda create -n sharktrack_venv anaconda
    conda activate sharktrack_venv
    pip install -r requirements.txt
    ```


#### Troubleshooting
Below you can find solutions to common problems encountered.
- *The terminal/prompt is showing an error saying I need `git`* Go ahead and [install it](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). Alternatively, download the latest "source code"  [from here](https://github.com/filippovarini/sharktrack/releases).
- *`python -m venv venv` says `python` not found* Try running `python3 -m venv venv`
- *`python -m venv venv` says `venv` module not found* Check the python version by running `python --version` and make sure it's older than 3.8.If not download it [here](https://www.python.org/downloads/). Once it's downloaded, try running the same command. If it still fails, run `python3 -m venv venv`

### Running the Model

You now are ready to run SharkTrack! 

The quickiest way to do so, is moving your videos in the `./input_videos` folder which is inside sharktrack.

Then, you can run the following command. It is fine if it takes some time to initialise!
```bash
python app.py
```

Alternatively, if you don't want to move the videos, you can tell SharkTrack to find them by running

```bash
python app.py --input <path_to_video_folder>
```

> After the model finishes running, it will show how many tracks it found! You can now check the results in the output folder. Now it is time to **Generate MaxN**. Move to the [next page for instructions](./annotation-pipelines.md).

#### Additional Arguments
The additional input arguments below provide additional functionality
- `--input` Path to the video folder. SharkTrack takes a folder of arbitrary depth as input and processes all .mp4 videos in it.
- `--stereo_prefix` If your folder contains Stereo-BRUVS, you can tell SharkTrack to only process the left or right video by passing the prefix of the videos you want to process (i.e. `LGX`)
- `--limit` Limit of videos to process (default=1000)
- `--output` Path to output folder (default=`./output`)
- `--mobile` Whether to run the mobile version of the model. More info [here](#mobile-vs-analyst). Example: `python app.py --mobile` 
- `--live` Boolean, pass it to output the same input video with a live detection annotation. Example: `python app.py --live`

## How fast is SharkTrack? Can I use it on my laptop?
We have provided 2 SharkTrack models, the mobile and analyst models. Both models are able to run on the CPU. The analyst model is more accurate, but takes more. 
You can find a more thorough comparison [here](#model-types-mobile-vs-analyst).

As a rule of thumb, we suggest running the more accurate model first. If that is too slow, you can switch to the mobile model by simply passing the `--mobile` in the [run script](#2-running-the-model).

## Model Types: Mobile vs Analyst
|Model|Accuracy (F1)| CPU Inference Time | Limitations | Good for
|--|--|--| --| --|
|`analyst`| 0.85 | 1.5x video speed | Can't process GoPro | Above-human-level detection accuracy
|`mobile`|0.83 | 3.5x video speed | Unstable tracking | Quick overview of daily BRUVS deployment

### GoPro Limitation 📹⛔️
The model uses `OpenCV` speed-up the video read and achieve creditable speed for its size. Unfortunately, OpenCV fails with the GoPro audio encoding (GoPro AAC), as documented [here](https://stackoverflow.com/questions/78039408/cv2-ffmpeg-grabframe-packet-read-max-attempts-exceeded-error-after-exactly-rea).

> Therefore, the model can't process GoPro videos

To solve this issue, we have provided a [script](./scripts/reformat_gopro.py) to reformat the videos by removing the audio stream. 

> **NOTE** You need to have `ffmpeg` installed. Check this by running `ffmpeg -version`. If you don't have it, download it [here](https://ffmpeg.org/download.html). We suggest following [this](https://www.youtube.com/watch?v=22vmzTs5BoE) tutorial.

You can run it with the following command:
```bash
python scripts/reformat_gopro.py --input {Original video folder path} --output {New video folder path}
```
If the videos are Stereo-BRUVS, you can use the `--stereo_prefix` to only reformat left/right videos, as described [here](#arguments).

This script takes approximately 6x the video speed. This is a time delay that we aim to remove in the future. 

The good news is that this script is an alternative command to copy data. Therefore, researchers can use it to transfer data from GoPro SD cards to the laptop/drive. In this case, the time delay will be the same as currently experienced doing data transfer.

#### So what should I do if I am doing GoPro BRUVS survey?
1. Collect BRUVS videos
2. When you get home, for each GoPro, connect the SD and run 
    ```bash
    python scripts/reformat_gopro.py --input PATH_TO_SD --output COPY_DESTINATION
    ```
    Ideally, use the same destination
3. Overnight run the model on the copied data 
    ```bash
    python app.py --input COPY_DESTINATION
    ```

If you know of a better solution, please [email us](mailto:fppvrn@gmail.com?subject=SharkTrackSuggestion)!

## Can I trust it's accuracy?
The `analyst` model outperforms human accuracy, which is at the level of the `mobile`, according to [Ditria et al 2020](https://www.frontiersin.org/articles/10.3389/fmars.2020.00429/full).

Additionally, the pipeline is designed to minimise False Positives and leverage human knowledge. That is, it has a very low confidence threshold. This ensures close to all sharks are detected, but causes 5x "garbage detections", which the user manually rejects. We found that researchers are happier to have control over the rejection of the detections, knowing that everything was captured.

That being said, sometimes SharkTrack doesn't detect an elasmobranch. In this case, the shark will be lost by the pipeline, causing unaccurate MaxN. However, our validation tests prove that the frequency is smaller than human error.

## Next steps
After following the steps you will have an output folder with detections. It is now time to remove the incorrect annotations and assign Species ID.

Please follow the documentations on [the next step](./annotation-pipelines.md).
