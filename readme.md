# SharkTrack
*Speeding up Elasmobranch BRUVS processing by 20x...*

## Contents

* <a href="#overview">Overview</a>
* <a href="#what-does-sharktrack-do">What does SharkTrack do?</a>
* <a href="#how-people-use-sharktrack-pipeline">How do people use SharkTrack?</a>
* <a href="#how-people-run-sharktrack">How do people run SharkTrack?</a>
* <a href="#list-of-people-using-it">List of people using it</a>
* <a href="#collaborations">Collaborations</a>

## Overview
Elasmobranch researchers monitor their populations using Baited Remote Underwater Video Systems (BRUVS). This is a time-consuming process, as each video needs to be manually annotated. 

SharkTrack is a Machine Learning model that uses computer vision to detect and track Elasmobranchii in BRUVS videos. The tool is designed to allow researchers to focus only on interesting video frames, removing the need of watching hour-long empty videos.

It integrates with commonly-used annotation tools to convert raw videos into detection and MaxN Metrics 20x faster than current workflows.

This page is for users who are just considering the use of AI in their workflow, and aren't even sure yet whether SharkTrack would be useful. It summarizes what we do with the model to help our collaborators who are overwhelmed by BRUVS videos to annotate. This page also includes some questions and information for new collaborators. 

If you're already familiar with SharkTrack and you're ready to run it on your data (and you have some familiarity with running Python code), see the [SharkTrack User Guide](./sharktrack-user-guide.md) for instructions on downloading and running the model.

[![watrch video](static/video_screenshot.png)](https://drive.google.com/file/d/1b_74wdPXyJPe2P-m1c45jjsV2C5Itr-R/view?usp=sharing)
*Click on the image above to watch a demo*

## What does SharkTrack do?
SharkTrack takes a folder with BRUVS videos and outputs something [like this](./static/test-output/). The output folder contains:
- `output.csv` lists each detection at each timeframe for each video
- `viame.csv` for each tracked shark, records the detection which achieved the highest confidence (`max-conf-detection`)
- `detections/` for each tracked shark, saves the `.jpg` frame in which the shark track achieved highest confidence.
    ![detection example](./static/test-output/detections/5.jpg)
    *The image shows the shark (red box) whose track achieved the higest confidence in this frame, over all others in which the same shark was detected. It also shows other detections (white boxes) and the video, time and confidence of the red detection*

The `max-conf-detections` extracted in `viame.csv` and `detections/` are used in the [annotation workflow](./annotation-pipelines.md) to allow the user to classify each shark only once per track and have the MaxN automatically computed from it. This tracker-enabled process is significantly faster than classifying each frame.

## How people use SharkTrack? (Pipeline)
SharkTrack alone is useless to researchers. We integrate the model into an end-to-end BRUVS processing pipeline that converts raw BRUVS into MaxN metrics.

Users of SharkTrack do so by:
1. Running the model over a batch of BRUVS videos [[Guide here]](./sharktrack-user-guide.md)
2. Removing incorrect detections and assigning species ID to the correct ones [[Guide here]](./annotation-pipelines.md)
4. Computing MaxN metrics [[Guide here]](./annotation-pipelines.md)

We have two main types of users:
### Field Researcher
They are currently doing BRUVS deployment. They want their daily videos to be processed overnight to share detections with the team and externally and have a precise idea of which sampling site records more sharks.

SharkTrack helps these users with the `mobile` model, which processes up to 25h of BRUVS overnight on a *CPU* without access to WIFI. The model prioritises speed over acuracy, so it is designed to obtain a quick overview, rather than accurate metrics.

### BRUVS Analyst
They analyse the BRUVS to compute accurate MaxN metrics. High accuracy is crucial to them, and they have less strict time constraints.

SharkTrack helps these users with the `analyst` model, which achieves higher accuracy than humans. The model prioritises accuracy over speed, and takes twice the time to run on *CPU*.

## How people run SharkTrack?
SharkTrack is a publicly-available model, and the [SharkTrack User Guide](./sharktrack-user-guide.md) provides instructions for running it using our Python scripts. Many of our users run SharkTrack on their own, either on the cloud or on their local computers.

That said, we know that Python can be a bit daunting to setup. Additionally, SharkTrack requires significant processing power which, despite we have designed it for mobile usage, might still be a challenge. Therefore, some users - particularly high-volume users - send us BRUVS videos (online or physical hard drives), which we run through SharkTrack, then we send back a results file. 

If that is of interest for you, please read the [Bespoke Deployment](###bespoke-deployment) section.

Whether you're going to run SharkTrack on your own or work with us, usually the first step with a new user is just running our model on a short BRUVS video and seeing what happens, so if you're interested in trying this on your BRUVS, we can work out a way to transfer a set of example images, just [email us](mailto:fppvrn@gmail.com?subject=SharkTrack-Pilot).

## List of people using it
- [University of Exeter and Government of Anguilla](https://www.linkedin.com/posts/filippo-varini_we-are-back-from-university-of-exeter-activity-7167899292593065985-dZLo?utm_source=share&utm_medium=member_desktop)

## Collaborations
### License & Citation
This repository is licensed with the [MIT license](https://opensource.org/license/mit). If you use MegaDetector in a publication, please cite:

Varini, F. et al (2024). SharkTrack. Github. Available at
https://github.com/filippovarini/sharktrack


The same citation, in BibTex format:

```BibTex
@article{varini2024sharktrack,
  title={SharkTrack},
  author={Filippo Varini et al},
  year={2024}
}
```
### Issues
Please submit any issue on [GitHub](https://github.com/filippovarini/sharktrack/issues). We aim to respond to it within a week.
### Contribution
This project welcomes contributions, as pull requests, issues, or suggestions by [email](mailto:fppvrn@gmail.com?subject=SharkTrackContribution).

This is the first step of a broader effort to develop generalisable marine species classifiers. We are looking for contributors for this project. If you want to get involved in AI-driven Ocean Conservation please email us.

### Contacts
- [Email](mailto:fppvrn@gmail.com?subject=SharkTrackGeneral)
- Website
- [Linkedin](https://www.linkedin.com/in/filippo-varini/)
- [X](https://twitter.com/filippo_varini)

### Bespoke Deployment
Sometimes our users have high volume of BRUVS videos they want to be processed or specific requirements for SharkTrack, such for it to classify certain species or work in particularly challenging environments. 

In this case, we offer a bespoke deployment service. We understand their requirements, collect their data and train a tailored model ourselves that satisfies their needs. Currently, we require funding for this service.

If this is of interest for your organisation, please reach out via email by answering the following questions:

1. Can you provide a short overview of your project? What ecosystem and locations are you working in? What is the water quality? What are the key species found (besides Elasmobranchii)?

2. About how many hours of BRUVS do you have waiting for processing right now? Are these stereo systems?

3. About how many hours do you expect to collect in the next, e.g., 1 year?

4. What tools do you use to process and annotate images? For example, do you:

    - Move images to folders named by species
    - Keep an Excel spreadsheet open and fill it with filenames and species IDs
    - Use a tool like Eventmeasure

5. About what percentage of your videos is empty?

6. If you are also interested in Species ID, what are the key speceis you would like to classify? 

7. Do you have an NVIDIA GPU available (or access to cloud-based NVIDIA GPUs)? "I don't know what a GPU is" is a perfectly good answer.

8. At the place where you plan to do most of your work, how is your bandwidth? If you're able to visit speedtest.net to measure your upload and download speeds, that's helpful.

9. How did you hear about SharkTrack?

10. Do you have any legal or policy constraints that prevent you from using cloud-based tools to manage or review your images?

11. What is your level of fluency in Python?

12. About how many images do you have that you've already annotated, from roughly the same environments? What annotations?

## Next Stpes
