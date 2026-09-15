# glowing-robot

Real-time facial recognition system built for a BWSI/CogWorks robotics capstone.

## What it is

A live webcam facial-recognition app: streams camera frames, detects and tracks faces with bounding
boxes in real time, and lets you interactively register new people by name into a small local face
database (`face_db.pkl`) from the terminal.

Built on course-provided libraries from CogWorksBWSI (`facenet_models`, `Camera`, `DataSets`),
kept here as git submodules rather than vendored copies — they are the program's teaching
libraries, not original code from this project.

## Stack

- Python, OpenCV, facenet-based face embeddings

## Credits

Built during a Beginning Workshop in Science and Innovation (BWSI) / CogWorks summer program, in
collaboration with teammate Hunter Baker.

## Status

The `Camera`, `DataSets`, and `facenet_models` submodules point at the course's original
repositories, which are no longer public. A fresh clone will need those three directories provided
separately before the code will run; `git submodule update --init` will not succeed against the
current submodule URLs.
