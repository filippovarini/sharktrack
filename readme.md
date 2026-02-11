<div align="center">

# SharkTrack

**AI-enhanced video analysis of sharks/rays to compute MaxN 21x faster.**


<img src="static/demo.gif" alt="SharkTrack Demo" width="720" />

[![Official Website](https://img.shields.io/badge/Official%20Website-00567a)](fvarini.com/sharktrack)

</div>

---

SharkTrack uses computer vision to detect and track elasmobranchs (sharks & rays) in BRUVS footage and compute species-specific MaxN.

> **For an in-depth guide**, including how the pipeline works and how to review results, visit the **[SharkTrack Website](https://www.fvarini.com/sharktrack)**.



## Quick Start

### 1. Download

Download the latest release from [GitHub Releases](https://github.com/filippovarini/sharktrack/releases) (zip file), unzip, and open a terminal in the `sharktrack` folder.

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
python app.py --input /path/to/your/videos
```

That's it! SharkTrack will process every `.mp4` in the folder and save detections to `./outputs`.

### Configuration

Customise the run with optional flags:

```bash
python app.py --input /path/to/videos --conf 0.25 --peek --stereo_prefix L --resume
```

| Flag | Description |
|---|---|
| `--input` | Path to video folder (processes all `.mp4` files recursively) |
| `--output` | Path to output folder (default: `./outputs`) |
| `--conf` | Detection confidence threshold (default: `0.25`) |
| `--peek` | Fast mode — extracts interesting frames only, no MaxN |
| `--stereo_prefix` | Only process videos whose filename starts with this prefix |
| `--chapters` | Aggregate chapter-split videos into one |
| `--resume` | Skip already-processed videos |
| `--limit` | Max number of videos to process (default: `1000`) |

### 4. Review & Compute MaxN

After processing, classify detections and generate MaxN. See the **[Annotation Pipeline Guide](https://www.fvarini.com/sharktrack/annotation-pipeline/)** for detailed instructions, or run:

```bash
python utils/compute_maxn.py --path outputs
```


## Using SharkTrack? Let Us Know!

SharkTrack is free and open-source. If you are using it in your research, we'd love to hear from you — it helps us improve SharkTrack and spotlight your work on our website.

<div align="center">

[![Tell us about your project](https://img.shields.io/badge/%F0%9F%93%8B%20Fill%20out%20the%20form-30%20seconds-brightgreen?style=for-the-badge)](https://forms.gle/sUeTarNqcTgxdyjU8)

</div>



## Citation

If you use SharkTrack, please cite:

```bibtex
@article{varini2024sharktrack,
  title={SharkTrack: an accurate, generalisable software for streamlining shark and ray underwater video analysis},
  author={Varini, F. and Gayford, J. H. and Jenrette, J. and Witt, M. J. and Garzon, F. and Ferretti, F. and Glocker, B.},
  journal={arXiv preprint arXiv:2407.20623},
  year={2024}
}
```


## Contributors

This software and related work was supported by the efforts of:

Filippo Varini, Joel H. Gayford, Jeremy Jenrette, Matthew J. Witt, Francesco Garzon, Francesco Ferretti, Sophie Wilday, Mark E. Bond, Michael R. Heithaus, Danielle Robinson, Devon Carter, Najee Gumbs, Vincent Webster, Ben Glocker, Fabio De Sousa Ribeiro, Rajat Rasal, Orlando Timmerman, Natalie Ng, Rui Wen Lim, Michael Sellgren, Lara Tse, Steven Chen, Maria Pia Donrelas, Manfredi Minervini, Xuen Bei (Bay) Chin, Adam Whiting, Aurora Crocini, Gabriele Bai, Stephanie Guerinfor.

## Support & Contact

- **Issues** — [Open a GitHub Issue](https://github.com/filippovarini/sharktrack/issues) (we aim to respond within a week)
- **Email** — [fppvrn@gmail.com](mailto:fppvrn@gmail.com?subject=SharkTrackHelp)
- **Contributions** — Pull requests, suggestions, and collaborations are welcome. This is part of a broader effort to develop generalisable marine species classifiers — [get in touch](mailto:fppvrn@gmail.com?subject=SharkTrackContribution) if you want to get involved in AI-driven ocean conservation.

---

<div align="center">
<sub>Licensed under the <a href="https://opensource.org/license/mit">MIT License</a></sub>
</div>
