from datetime import datetime
from pathlib import Path
import shutil
import click

def get_date():
    now = datetime.now()
    date_string = now.strftime("%d_%m_%y")
    return date_string


def create_folder(**args):
    path = Path(args["drive"]) / args["study"]
    path = path / args["location"] / args["day"]
    path = path / f"lance_{args['launch']}"
    path = path / f"station_{args['station']}"
    path = path / args["side"]

    try:
        path.mkdir(parents=True)
    except:
        print("path already exists")
        return None
    return path

def copy_videos(**args):
    folder_path = create_folder(**args)
    if not folder_path:
        return

    sd_path = Path(args["sd_path"])
    for f in sd_path.rglob("*.[Mm][Pp]4"):
        if not f.name.startswith("."):
            print(f"Copying video {f}")
            new_path = folder_path / f"{args['side']}{f.name}"
            shutil.copy(str(f), str(new_path))

@click.command()
@click.option("--sd_path", "-sd", type=str, prompt="Path of SD card")
@click.option("--drive", "-d", type=str, prompt="Path of Hard Drive")
@click.option("--day", type=str, default=None)
@click.option("--launch", type=str, prompt="Launch")
@click.option("--station", type=str, prompt="Station")
@click.option("--side", type=str, prompt="Side")
def main(**kwargs):
    kwargs["study"] = "OBT"
    kwargs["location"] = "CABO_PEARCE"
    kwargs["day"] = kwargs["day"] or get_date()
    copy_videos(**kwargs)


if __name__ == "__main__":
    main()
    
    