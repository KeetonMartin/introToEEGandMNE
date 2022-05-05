# Introdution to EEG and MNE+Python
For my independent study on EEG Data Analysis with Python and MNE.
This file can serve in part as a user manual, and a guide to the scripts and data in this repository.

## Contents:
1. General thoughts
3. Basics of how to use the terminal
1. Installations necessary for project
2. Basics of how to use github and git
4. Basics of how to run python from terminal

## General Thoughts
This guide will assume that you're using a Mac / Linux machine. While most of this stuff should work on windows, it's hard for me to know for sure what will work where because I'm on a mac myself. Note that google is your friend when things don't work throughout this project, as is the MNE documentation. 

MNE Workshop from Mainak, and some good documentation: https://github.com/mne-tools/mne-workshops/tree/master/2019_04_Brown
MNE Official Documentation (Tutorial Page): https://mne.tools/stable/auto_tutorials/intro/10_overview.html#sphx-glr-auto-tutorials-intro-10-overview-py
MNE "Advanced" Install via Pip: https://mne.tools/stable/install/manual_install.html
Internet Thoughts on Mac vs. Windows Command Line: https://www.quora.com/What-are-the-differences-between-the-Mac-Terminal-and-the-Windows-Command-Line

## Navigating with the Terminal
When you open your terminal, you should see something like this, which is known as a command prompt:
```console
foo@bar:~$
```
This is known as the command line / terminal, the terms are basically used interchangably. You can run commands here. We're going to try a few out. Type this command "whoami" into your command line and press enter.
```console
foo@bar:~$ whoami
foo
```
You can do almost anything that you can do in Finder using the terminal instead. This is useful especially when you're using your terminal to interact over a network with other "machines" (computers). For us, we're doing it for fun to navigate, but eventually, we'll also be using the terminal to run our files. There **are** other ways to do this stuff, but we're going to use the terminal for the sake of consistency. 

> **_NOTE:_**  "directory" is just another word for folder.

We're also going to need the following commands:
- `pwd` to view the "path to the working directory"
- `ls` to view the contents of the current directory
- `cd` stands for change director. We can use it to go into a dir or "back up" to the parent folder.
    - `cd subFolder`
    - `cd ..`
- `mkdir newFolder` to create a new dir in your current working directory, named newFolder.

## Installations

### Python
We can also use certain terminal commands to access programming languages. We'll start by checking the current installed version of python:
```console
Keetons-Computer@Keeton:~$ python --version
Python 3.8.2
```
As long as your version is greater than 3.0.0 you should be fine for this project. If you need to update: https://www.python.org/downloads/macos/

### MNE
MNE has their own installation guide but keeping things simple you should be fine to just run
```console
Keetons-Computer@Keeton:~$ pip install mne
```

To get pip on windows: curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

## Basis of Git and Github
Now what you're going to want to do is actually get this code in this repo downloaded to your machine. The easiest way to do this is with `git clone`.
Git is what's known as a Version Control System. You can think of it like google docs for code. Git comes installed on macs and linux machines so you should be good to run these commands. But before we download the code, we need to navigate to a directory that you want to put everything in. For me, I like to keep stuff in my `Documents` folder, but some people like to keep stuff in their `Desktop` or other places on their machine.

For me, my folder I want to clone into is `/Users/keeton/Documents/courses/neuro403/`
Once you're ready to clone into your `wd`, run the following
```console
Keetons-Computer@Keeton:~$ git clone https://github.com/KeetonMartin/introToEEGandMNE.git
```
Once this command finishes, you should be ready to run some python code!

## Run Python from the Terminal
Now what you've got your code downloaded, let's talk about running it. There are multiple different ways to run python code. The basic one that I think is going to be the simplest to figure out is to run the `python file.py` command from the terminal followed by the name of the file you want to run. This will run as long as `file.py` is in your current `wd`. You can also run python files outside the `wd` if you specify the path. In this example, I'd like you to `cd singleSubjectPipeline` so that the python file we're about to run will look for data in the right place.

So to get started, try running
```console
Keetons-Computer@Keeton:~$ python singleSubjectPipeline/singleSubject.py
```
If you start getting output (non-errors), then you're probably good to go. The output should ask for input at certain points, and you might just have to press enter to make it continue. 

## Conclusion
That's about it for this guide / tutorial for now. If you're getting consistent errors / bugs, feel free to file a `Github Issue`, or fork and try to fix it yourself. 
