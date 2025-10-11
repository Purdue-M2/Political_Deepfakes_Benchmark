# Code for generated image detection and generated image attribution

# Requries GPU

1. download the models from [here](https://drive.google.com/file/d/1GEFSmua_4VkBV1_MU0mi5rJ7BAbcuDtv/view?usp=sharing)
2. untar, so that models folder is placed in your code before a docker build

# For docker build (and to avoid complexity of local install)

1. docker build -t ganattribution  .

2. To set the data mount for images you want to process: DATA=your_local_folder_with_images

3. docker run  -v $DATA:/mnt/ -it   ganattribution   bash

or

4. docker run --rm  -v $DATA:/mnt/ -it   ganattribution   bash

# For local install

Other packages please run:
```
pip install -r requirements.txt
```


# python test_general.py

provides examples for scoring single images as synthetics or pristine as well as synthesized image attribution into six types

# License 

MIT License (MIT)


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

