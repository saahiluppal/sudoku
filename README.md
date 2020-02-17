# Virtual Sudoku Solver
This project aims to solve SUDOKU puzzles from papers in your hand.

## Implementation:
Each frame taken from webcam will be converted to grayscale and then edges are to be detected. Then dilation is performed on the grayscaled image followed by erosion to remove noise from the grayscaled image. Now we have neat and clean grayscale image. After that Sudoku grid lines are detected and filtered so that 2 or more detected lines representing same original line have to be preprocessed. After this intersection points are detected where these lines intersect each other in each grayscaled frame. Now we have discrete boxes in which numbers are located or may not be located (which are to be found [null boxes]). Then iterating over each box we predict whether it have a number or it's a null box. if it have a number we have find it's actual rank. and Finally sudoku puzzle is solved using backtracking.

## Requirements:
- Tensorflow >= 2.0.0
- Opencv >= 2.4
- Numpy

## Usage:
```bash
$ python main.py
```

## LICENSE:
<a href='https://github.com/saahiluppal/sudoku/blob/master/LICENSE'>Apache License 2.0</a>
