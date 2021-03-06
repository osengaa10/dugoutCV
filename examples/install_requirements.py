#!/usr/bin/env python3
import sys, os, subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-sdai', "--skip_depthai", action="store_true", help="Skip installation of depthai library.")
args = parser.parse_args()

examples_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(parent_dir)
sys.path.insert(1, parent_dir)
import find_version

# 3rdparty dependencies to install
DEPENDENCIES = ['opencv-python', 'pyyaml', 'requests']

# Constants
ARTIFACTORY_URL = 'https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local'

# Check if in virtual environment
in_venv = getattr(sys, "real_prefix", getattr(sys, "base_prefix", sys.prefix)) != sys.prefix
pip_call = [sys.executable, "-m", "pip"]
pip_install = pip_call + ["install"]
if not in_venv:
    pip_install.append("--user")

# Update pip
subprocess.check_call([*pip_install, "pip", "-U"])
# Install opencv-python
subprocess.check_call([*pip_install, *DEPENDENCIES])

if not args.skip_depthai:
    # Check if in git context and retrieve some information
    git_context = True
    git_commit = ""
    git_branch = ""
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('UTF-8').strip()
        git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('UTF-8').strip()
    except (OSError, subprocess.CalledProcessError) as e: 
        git_context = False

    # Install depthai depending on context
    if not git_context or git_branch == 'main':
        # Install latest pypi depthai release
        subprocess.check_call([*pip_install, '-U', '--force-reinstall', 'depthai'])
    elif git_context:
        try:
            subprocess.check_output(['git', 'submodule', 'update', '--init', '--recursive'])
        except (OSError, subprocess.CalledProcessError) as e: 
            print("git submodule update failed!")
            raise
        # Get package version if in git context
        final_version = find_version.get_package_dev_version(git_commit)
        # Install latest built wheels from artifactory (0.0.0.0+[hash] or [version]+[hash])
        commands = [[*pip_install, "--extra-index-url", ARTIFACTORY_URL, "depthai=="+final_version],
                    [*pip_install, "."]]
        success = False
        for command in commands:
            try:
                success = subprocess.call(command) == 0
            except (OSError, subprocess.CalledProcessError) as e:
                success = False    
            if success:
                break

        # If all commands failed
        if not success:
            print("Couldn't install dependencies as wheels and trying to compile from sources failed")
            print("Check https://github.com/luxonis/depthai-python#dependencies on retrieving dependencies for compiling from sources")

subprocess.check_call([sys.executable, f"{examples_dir}/downloader/downloader.py", "--all", "--cache_dir", f"{examples_dir}/downloader/", "--num_attempts", "5", "-o", f"{examples_dir}/models"])
