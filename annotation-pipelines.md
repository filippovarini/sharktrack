# Annotation Pipeline
*From model output to species MaxN metrics...*

## Contents

* <a href="#overview">Overview</a>
* <a href="#understand-the-output">Understand the Output</a>
* <a href="#cleaning-the-output">Cleaning the output</a>
* <a href="#extract-maxn">Extract MaxN</a>

## Overview
After running SharkTrack on your videos, you should have an output similar to [this folder](./static/test-output/).

This page provides instruction on generating MaxN metrics from the model output. To do so, you would need to perform two steps:
1. **Clean** the model output: delete wrong detections and assign species ID to the correct ones
2. Generate **MaxN** from the cleaned output

You can also follow this documentation using the following video tutorials:

- [Uploading and cleaning detections in VIAME](https://drive.google.com/file/d/16Zw69ELvA1_pBhfcbQsjo1nc_7EBYZl2/view?usp=sharing)
- [Computing MaxN after downloading VIAME-cleaned detections](https://drive.google.com/file/d/1DCT3vCAbAH4T8wTiMjgWUc7-lZEpgz9U/view?usp=drive_link)

## Understand the Output
Before we compute MaxN, it is important to understand the model output.

> **TL;DR:** When SharkTrack detects a shark, it tracks it across frames, generating a "Shark Track". Then, it saves all results in `output.csv` and for each track stores in `detections/` a frame where it was detected. Thanks to this, the user only needs to validate/classify the same shark once as we take care of tracking it!

Locate the output directory. It should be `./output`, unless you have provided a custom `--output_dir` argument. The folder will look something [like this](./static/test-output/) and it contains :
- `output.csv` lists each detection at each timeframe for each video
- `viame.csv` annotations in VIAME format, to integrate with the VIAME annotation tool (more below) 
- `detections/` for each tracked elasmobranch, saves the `.jpg` frame in which its track achieved highest confidence.

    <img src="./static/test-output/detections/14.jpg" width=600 />

    *The image shows the shark (red box) whose track achieved the higest confidence in this frame, over all others in which the same shark was detected. It also shows other detections (white boxes) and the video, time and confidence of the red detection*

### Output FAQs
- **What is a track?** The same elasmobranch will appear in multiple (consecutive) frames of the video. A track is a bounding box with an id, saying "this is the same shark I found before" 
- **What is a `max-conf-detection`?** A track is made up of different detections for the same elasmobranch at different times. Each detection has a confidence score. The `max-conf-detection` is the detection (frame,time,bounding box) which achieved the highest score, and it's associated frame is saved in the `./detections/` folder for each track
- **Why does it matter?** This allows the user to process only once annotation per track, instead of thousands of frames. Once the user is done with the cleaning, a script automatically reflects the changes on every detection for the track, computing MaxN
- **But one frame is not enough to determine the species of a shark** That's why we show in the frame the video path and time, so you can go back to the video, and use it to assign a Species.
- **I am seing the same elasmobranch in multiple detections**: The model might lose the track and create a new one. This will not affect the MaxN accuracy, it will only require you to clean more images.
- **But...?** If you have any other question, feel free to [email us](mailto:fppvrn@gmail.com?subject=SharkTrackFAQ)


## Cleaning the Output
Here, we define *detections* as the image showing the highest-confidence detection of a Track (*same elasmobranch in multiple frames*). The model outputs detections in [this format](./static/test-output/detections/).

Now that you understand the model output, you need to clean it by:
- Rejecting tracks that are not of elasmobranchii
- Assigning species ID to the correct ones left

### Cleaning Guide
Each track has an associated image in `detections/` for you to clean it.

When you look at a detection image, you are cleaning the detection in the red box. The white boxes are there only to provide context and their tracks are saved in other detection files for you to clean. 

<img src="./static/test-output/detections/8.jpg" width=400 />

*In this image refers to the detection in the red box (bottom right). By classifying this image, you are classifying only that detection*

Below, we provide examples of detection files and what cleaning action should be performed


| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./static/test-output/detections/15.jpg" >  Confirm detection and assign species |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./static/test-output/detections/16.jpg" /> Reject detection as already tracked (white box) |<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./static/test-output/detections/14.jpg" > Confirm detection and assign species |
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./static/test-output/detections/1.jpg" > Confirm and assign species |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./static/test1.jpg"> Reject as it already exists (white box)|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./static/test2.jpg"> Reject detection as not elasmobranch|
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./static/test3.jpg"> Confirm and assign ray species  |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./static/test4.jpg"> Definitely not elasmobranch!|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./static/test5.jpg"> Hmm tricky! Check the suggested video and time to know better!|

### Pipelines
SharkTrack proposes two ways of cleaning the output:
|Cleaning Method|Description|When to Use it| Guide|
|--|--|--|--|
|Local| Using your laptop, delete wrong detections and rename valid with the species | Very quick and intuitive, don't need wifi or third-party app | [here](#local-user-guide)
|VIAME|  Upload and clean detections on the [VIAME](https://viame.kitware.com/) annotation tool | Allows multiple analysts to collaborate | [here](#viame-user-guide)

Please check the respective guide for instructions.

> ⏰ As a rule of thumb, you should be able to clean 50 detections per minute. 100h of BRUVS should prodice ~3000 detections -> just 1h of cleaning!

### Local User Guide
1. Open the `detections` folder ([example](./static/test-output/detections/)). It will have many detections names `{track_id}.jpg`
2. Scroll through all images and locate the relative detection, which is the red box
3. If the detection is not an elasmobranch, delete the file
4. If the detection is an elasmobranch, rename the file to `{track_id}-{species_id}.jpg`
    
    **Important:** you can use whichever species_id but make sure to keep the original `track_id`, and separate it with a "-"

You can find a tutorial [here]

<img src="./static/local-cleaning.png" width=500/>

#### 🚀 Pro Tips
- Do a first pass to remove all wrong detections and assign species ID in a second pass
- If unsure about species/validity, use the image bottom text to find the relative video and time.
- If on Mac, visualise image in Gallery mode, use Cmd+Del to remove the image and Enter to rename the file

### VIAME User Guide

        
#### Setup Annotations Platform
1. Open [VIAME](https://viame.kitware.com/)
2. Create an account
3. Click “Upload“ > Add Image Sequence
    
    <img src="static/Screenshot_2024-03-15_at_16.45.25.png" width=400 />
    
4. Upload all the images in `./detections`
5. Click on “annotation file” and upload `viame.csv`
    
    <img src="static/Screenshot_2024-03-15_at_16.47.14.png" width=400/>
    
6. Pick a name for the BRUVS analysis

    <img src="static/analysis_name.png)
7. Confir" width=400/> upload
#### Clean Annotations
1. Click Launch Annotator
2. For each frame
    
    <img src="static/Screenshot_2024-03-15_at_16.49.38.png" width=400/>
    
    i. Identify the track by clicking on the highlighted bounding box

    ii. If the detection is valid, insert the shark species
        
    <img src="static/Screenshot_2024-03-15_at_16.52.17.png" width=400/>
        
    iii. If the detection is invalid, delete the track by clicking on the trash
        
    <img src="static/Screenshot_2024-03-15_at_16.53.25.png" width=400/>
            
#### Download Cleaned Annotations
    
<img src="static/Screenshot_2024-03-15_at_16.54.04.png" width=400 />

1. Save the changes by click on the 💾 Icon
2. Then click Download > "VIAME CSV" and download the file


#### 🚀 Pro Tips
- Navigate with the top/down arrows between frames
- Press the "Delete" key to delete garbage detection
- Press Shift+Enter to assign a species to the detection
Collaboration:
- Other users can contribute by searching your username in the VIAME search bar and locating your folder
- You need to give them access by right clicking on the project > Access Control
- You can save the changes by click on the 💾 Icon and resume later

## Extract MaxN
Amazing! You have cleaned all annotations, it's time to generate MaxN from it!

- Open the Terminal and move to the `sharktrack` repo that you installed in [this step](./sharktrack-user-guide.md#1-environment-setup).
- Activate the virtual environment (guide [here](./sharktrack-user-guide.md#1-environment-setup))
- Run `python scripts/compute_maxn.py --output {model_output}`
    - **NOTE** If you used VIAME to clean detections, you will need to pass the downloaded cleaned annotations. You can do this by running instead: `python scripts/compute_maxn.py --output {model_output} --viame_cleaned {downloaded_viame.csv}`
- You will see a `maxn.csv` file in the SharkTrack folder

🚀 Hooray! You have obtained the MaxN! 