# Installation

#sudo apt-get install libhdf5-10 libhdf5-serial-dev libhdf5-dev libhdf5-cpp-11
#sudo apt install libopenexr-dev zlib1g-dev openexr
#sudo apt install xorg-dev  # display widows
#sudo apt install libglfw3-dev

conda env create --file environment_cleargrasp.yml
conda activate cleargrasp

cd api/depth2depth/gaps/
find /usr -iname "*hdf5.h*"
export CPATH="/usr/include/hdf5/serial/"
make
bash depth2depth.sh
cd ../../../

# Get dataset

cd data
wget http://clkgum.com/shreeyak/cleargrasp-checkpoints
wget clkgum.com/shreeyak/cleargrasp-dataset-test
wget http://clkgum.com/shreeyak/cleargrasp-dataset-train

unzip cleargrasp-checkpoints.zip
tar -xf cleargrasp-dataset-test-val.tar
tar -xf cleargrasp-dataset-train.tar
cd ..
