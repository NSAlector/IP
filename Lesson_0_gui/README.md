# argus_test

Do the following steps to use  argus_test application.


# 1.Install ImageMagick

> sudo apt update 

> sudo apt-get install libpng-dev

> sudo apt-get install libjpeg-dev

> sudo apt-get install build-essential

> wget https://www.imagemagick.org/download/ImageMagick.tar.gz

> tar xvzf ImageMagick.tar.gz

> cd ImageMagick-7.0.8-27/

> ./configure --with-modules

> make

> sudo make install 

> sudo ldconfig /usr/local/lib

# 2. Build library and executable files for argus_test
> cd <path-to-"argus_test_project/CMakeList.txt">

> mkdir build

> cd build/

> cmake ..

> make

# 3. Enjoy! :)

> ./argus_test
