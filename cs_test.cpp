#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <thread>

#include "fft.h"

#define IMAGE_WIDTH  960
#define IMAGE_HEIGHT 640


int main() {


    TwoDCArray image (IMAGE_HEIGHT, CArray(IMAGE_WIDTH));
    readJPEG("flower.jpg", image, IMAGE_WIDTH, IMAGE_HEIGHT);


    fft2D(image, {}, 5);
    

    return 0;
}