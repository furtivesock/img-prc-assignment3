# Image processing assignment 3

Source code for the third assignment in Image processing course at Polytech Paris-Saclay engineering school.

- [Image processing assignment 3](#image-processing-assignment-3)
  - [Introduction](#introduction)
  - [Prerequisites](#prerequisites)
  - [File tree](#file-tree)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Exercise 1](#exercise-1)
    - [Exercise 2](#exercise-2)

## Introduction

In this assignment, we are trying to guess the placement (x, y, rotation) of fragments on the two following fresco :

***The creation of Adam* fresco by Michelangelo :**
![Alt text](Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg?raw=true "Michelangelo - The creation of Adam fresco")

Example of fragments :

![Alt text](Michelangelo/frag_eroded/frag_eroded_0.png?raw=true "Michelangelo - fragment 1")
![Alt text](Michelangelo/frag_eroded/frag_eroded_9.png?raw=true "Michelangelo - fragment 10")
![Alt text](Michelangelo/frag_eroded/frag_eroded_99.png?raw=true "Michelangelo - fragment 100")

**Virgin an unicorn by Domenichino :**

![Alt text](Domenichino_Virgin-and-unicorn/Domenichino_Virgin-and-unicorn.jpg?raw=true "Virgin an unicorn by Domenichino fresco")

Example of fragments :

![Alt text](Domenichino_Virgin-and-unicorn/frag_eroded/frag_eroded_18.png?raw=true "Michelangelo - fragment 10")
![Alt text](Domenichino_Virgin-and-unicorn/frag_eroded/frag_eroded_99.png?raw=true "Michelangelo - fragment 100")

## Prerequisites

- Python 3
  - OpenCV module

## File tree

```sh
img-prc-assignment3
├── Domenichino_Virgin-and-unicorn
│   ├── Domenichino_Virgin-and-unicorn.jpg # Original fresco
│   └── frag_eroded/ # Folder containing fragments
├── Michelangelo
│   ├── Michelangelo_ThecreationofAdam_1707x775.jpg # Original fresco
│   ├── frag_eroded/ # Folder containing fragments
├── ex1_interest_points/ # List of key points extraction test script
├── ex1_matching/ # List of key points matching test script
├── ex1_1_fragments_associations.py # kp extraction demo
├── ex1_2_fragments_placement.py # kp matching demo
├── ex2.py # Our final fresco reconstitution attempt
├── README.md
└── utils.py
```

## Installation

Clone the project

```sh
git clone https://github.com/furtivesock/img-prc-assignment3.git
cd img-prc-assignment3/
```

## Usage

### Exercise 1

You can visualize, our test script for key point extraction and matching :

**Key points :**
```sh
python3 ex1_interest_points/Method1_Harris.py
python3 ex1_interest_points/Method2_shi-tomasi.py
python3 ex1_interest_points/Method3_SIFT.py
python3 ex1_interest_points/Method4_SURF.py # Does not work
python3 ex1_interest_points/Method5_FAST.py
python3 ex1_interest_points/Method6_ORB.py
```

**Matching :**

```sh
python3 ex1_matching/Method1_BfMatcher.py
python3 ex1_matching/Method2_knnMatch.py
python3 ex1_matching/Method3_FlanBased.py # Does not work
```
The `ex1_1_fragments_associations.py` script will show the 2 best matches for each fragment. At the end of the script will be displayed the number of fragments that got enough key points and enough matches

```sh
python3 ex1_1_fragments_associations.py
```

The `ex1_2_fragments_placement.py` script will try to place each fragment on the original fresco _without taking the fragment rotation into consideration_.

```sh
python3 ex1_2_fragments_placement.py
```

### Exercise 2

The `ex2.py` script will try to place each fragment on the original fresco _taking the fragment rotation into consideration_. It does it using the RANSAC algorithm.

```sh
python3 ex2.py
```
At the end of the script will be displayed the result, here is an example with the Michelangelo fresco :

![Alt text](result_Michelangelo_10000ite.png?raw=true "Result exemple")

A `stats.prof` file will be generated in the root folder. It can be visualized with snakeviz :

![Alt text](stats.png?raw=true "Performance stats")