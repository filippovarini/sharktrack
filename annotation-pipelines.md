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


## Understand the Output
Before we compute MaxN, it is important to understand the model output.

> **TL;DR:** When SharkTrack detects a shark, it tracks it across frames, generating a "Shark Track". Then, it saves all results in `output.csv` and for each track stores in `detections/` a frame where it was detected. Thanks to this, the user only needs to validate/classify the same shark once as we take care of tracking it!

Locate the output directory. It should be `./output`, unless you have provided a custom `--output` argument when running `python app.py`. The folder will look something [like this](./static/test-output/) and it contains :
- `output.csv` lists each detection at each timeframe for each video
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
SharkTrack saves an image for each track in the `detections/` folder. In this image, it highlights the track with a *thick* bounding box. Additionally, it also tells what else it found in the same image, with *thin* bounding boxes. 

The purpose of the *thin* boxes is for you to that the model missed a shark if there is no thick or thin box around it and thus be confident about the weaknesses of it.

<img src="./static/test-output/detections/8.jpg" width=400 />

*In this image refers to the detection in the thick box (bottom right). By classifying this image, you are classifying only that detection*

Sometimes, it is ambiguous whether you should accept or reject the detection as invalid one. Therefore, below, we provide examples of detection files and what cleaning action should be performed.


| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./static/test-output/detections/15.jpg" >  Confirm detection and assign species |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./static/test-output/detections/16.jpg" /> Reject detection as already tracked (white box) |<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./static/test-output/detections/14.jpg" > Confirm detection and assign species |
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./static/test-output/detections/1.jpg" > Confirm and assign species |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./static/test1.jpg"> Reject as it already exists (white box)|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./static/test2.jpg"> Reject detection as not elasmobranch|
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./static/test3.jpg"> Confirm and assign ray species  |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./static/test4.jpg"> Definitely not elasmobranch!|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./static/test5.jpg"> Hmm tricky! Check the suggested video and time to know better!|

### Annotation Pipeline
Once you have seen the output, it is time to clean it. This is the step where we leverage *your* knowledge to ID the Elasmobranch.

1. Open the `detections` folder ([example](./static/test-output/detections/)). It will have many detections named `{track_id}.jpg`
2. Scroll through all images and locate the relative detection, which is the *thick* box
3. If the detection is not an elasmobranch, delete the file
4. If the detection is an elasmobranch, rename the file to `{track_id}-{species_id}.jpg`
    
    **Important:** you can use whichever species_id but make sure to keep the original `track_id`, and separate it with a "-"


<img src="./static/local-cleaning.png" width=500/>
*As shown in this picture, we are renaming the detection with the species id, in this case "ray". We are leaving the original track_id and separating with a "-".

#### 🚀 Pro Tips
- Do a first pass to remove all wrong detections and assign species ID in a second pass
- If unsure about species/validity, use the image bottom text to find the relative video and time.
- If on Windows, use F2 to rename the file quickly and Crl+D to remove it
- If on Mac, visualise image in Gallery mode, use Cmd+Del to remove the image and Enter to rename the file

#### Collaborating
If you want multiple users to annotate the detections, you can simply upload the whole `./detections` folder on a drive (Google, DropBox, OneDrive etc) and perform the above steps there!


## Extract MaxN
Amazing! You have cleaned all annotations, it's time to generate MaxN from it!

- Open the Terminal and move to the `sharktrack` repo that you installed in [this step](./sharktrack-user-guide.md#1-environment-setup).
- Activate the virtual environment (guide [here](./sharktrack-user-guide.md#setup-environment))
- Run `python scripts/compute_maxn.py --path {path_to_model_cleaned_results}`
    - **NOTE** Here `path_to_model_cleaned_results` is the path to the model output (default `./output`) where inside you have the `output.csv` file and the `detecrtions` folder, with cleaned detections and assigned Species ID
- You will see a `maxn.csv` file in the SharkTrack folder

🚀 Hooray! You have obtained the MaxN! 