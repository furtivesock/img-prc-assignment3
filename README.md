# Image processing assignment 3

Source code for third assignment in Image processing course at Polytech Paris-Saclay engineering school.

- [Image processing assignment 3](#image-processing-assignment-3)
  - [Introduction](#introduction)
  - [Prerequisites](#prerequisites)
  - [File tree](#file-tree)
  - [Installation](#installation)

## Introduction
In this assignment, we are tring to gess the placement (x, y, rotation) of fragments on the two folowing fresco :

**The creation of Adam fresco by Michelangelo :**
![Alt text](Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg?raw=true "Michelangelo - The creation of Adam fresco")

Exemple of fragments :
![Alt text](Michelangelo/frag_eroded/frag_eroded_0.png?raw=true "Michelangelo - fragment 1")
![Alt text](Michelangelo/frag_eroded/frag_eroded_9.png?raw=true "Michelangelo - fragment 10")
![Alt text](Michelangelo/frag_eroded/frag_eroded_99.png?raw=true "Michelangelo - fragment 100")

**Virgin an unicorn by Domenichino :**
![Alt text](Domenichino_Virgin-and-unicorn/Domenichino_Virgin-and-unicorn.jpg?raw=true "Virgin an unicorn by Domenichino fresco")

Exemple of fragments :
![Alt text](Domenichino_Virgin-and-unicorn/frag_eroded/frag_eroded_18.png?raw=true "Michelangelo - fragment 10")
![Alt text](Domenichino_Virgin-and-unicorn/frag_eroded/frag_eroded_99.png?raw=true "Michelangelo - fragment 100")

## Prerequisites

- Python 3
  - OpenCV

## File tree

```sh
img-prc-assignment3
├── Domenichino_Virgin-and-unicorn
│   ├── Domenichino_Virgin-and-unicorn.jpg # Original fresco
│   └── frag_eroded/ # folder containing fragments
├── Michelangelo
│   ├── Michelangelo_ThecreationofAdam_1707x775.jpg # Original fresco
│   ├── frag_eroded/ # folder containing fragments
│   ├── fragments_s.txt
│   └── fragments.txt
├── ex1_interest_points/ # List of key point exctraction test script
├── ex1_matching/ # List of key point matching test script
├── ex1_1_fragments_associations.py # kp exctraction demo
├── ex1_2_fragments_placement.py # kp matching demo
├── ex2.py # Our final fresco reconstitution attempt
├── README.md
└── utils.py
```

## Installation

1. Clone the project first

```sh
git clone https://github.com/furtivesock/img-prc-assignment3.git
```