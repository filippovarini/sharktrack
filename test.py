from argparse import ArgumentParser

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--stereo", action='store_true', help="Whether folder contains stereo BRUVS (LGX/RGX)")
  print(parser.parse_args().stereo)
