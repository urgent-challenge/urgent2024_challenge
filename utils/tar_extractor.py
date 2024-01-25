import argparse
import math
import shutil
import sys
import tarfile
import tqdm
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Extract the content of a tar file into a directory, but also creates
        sub-folder so that the total number of files in each sub-directory does not
        exceed a maximum"""
    )
    parser.add_argument("-i", "--input", type=Path, help="The tar file to extract")
    parser.add_argument(
        "-o", "--output_dir", type=Path, help="The directory to extract to"
    )
    parser.add_argument(
        "-m",
        "--max_files",
        type=int,
        default=10000,
        help="The maximum number of files per sub-directory",
    )
    parser.add_argument(
        "-p",
        "--pipe",
        action="store_true",
        help="Pipe the tar file from standard input",
    )
    parser.add_argument(
        "--file_limit",
        type=int,
        default=None,
        help="Limit the total number of files to extract",
    )
    parser.add_argument(
        "-d",
        "--num_digits",
        type=int,
        default=6,
        help="Number of digits in the subdirectory name",
    )
    parser.add_argument(
        "--target_suffix",
        type=str,
        nargs="+",
        default=[".wav", ".flac", ".mp3"],
        help="Suffix of the files that need to be divided into subfolder",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip existing files",
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        help="Skip files that produce an error",
    )
    args = parser.parse_args()

    # Create the output directory if it does not exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        tf_kwargs = {"name": args.input, "mode": "r:*"}
    elif args.pipe:
        tf_kwargs = {"fileobj": sys.stdin.buffer, "mode": "r|*"}
    else:
        raise ValueError("Either input or pipe must be specified")

    num_digits = args.num_digits
    file_number = 0
    dir_number = 0

    # Open the tar file
    with tarfile.open(**tf_kwargs) as tar:
        # Iterate over all the files in the tar file
        idx = 0
        t = tqdm.tqdm()
        while True:
            try:
                member = tar.next()
            except tarfile.ReadError:
                # if there is a read error, there are likely no more files ahead
                break

            if member is None:
                break

            if args.file_limit is not None and idx == args.file_limit:
                break
            idx += 1

            # Correct the path
            path = member.name
            if path.startswith("/"):
                path = path[1:]
            path = Path(path)

            if path.suffix not in args.target_suffix:
                # extract to regular directory
                try:
                    tar.extract(member, args.output_dir)
                except tarfile.ReadError as e:
                    if args.skip_errors:
                        continue
                    else:
                        raise e

            else:
                # insert sub-directory into the path
                name = path.name
                subdir = f"{dir_number:0{num_digits}x}"
                path = path.parent / f"{subdir}/{name}"
                output_path = args.output_dir / path

                # create the sub-directory if it does not exist
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if not (output_path.exists() and args.skip_existing):
                    # Extract the file
                    # we don't use the tar.extract method because it does not allow
                    # to rename the file as we want
                    try:
                        f = tar.extractfile(member)
                        if f is None:
                            # we are skipping things that are not files/links
                            continue
                        with open(output_path, "wb") as fout:
                            shutil.copyfileobj(f, fout)
                    except tarfile.ReadError as e:
                        if args.skip_errors:
                            continue
                        else:
                            raise e

                file_number += 1
                if file_number == args.max_files:
                    file_number = 0
                    dir_number += 1

            t.update(n=1)
