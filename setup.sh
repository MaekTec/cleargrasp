# Installation

# System Requiremetns
#sudo apt-get install libhdf5-10 libhdf5-serial-dev libhdf5-dev libhdf5-cpp-11
sudo apt install libopenexr-dev zlib1g-dev openexr
#sudo apt install xorg-dev  # display widows
#sudo apt install libglfw3-dev

# Conda env
conda env create --file environment_cleargrasp.yml
conda activate cleargrasp

# Build global optimization
cd api/depth2depth/gaps/
find /usr -iname "*hdf5.h*"
export CPATH="/usr/include/hdf5/serial/"
make
bash depth2depth.sh
cd ../../../

# Get dataset
cd data
wget http://clkgum.com/shreeyak/cleargrasp-checkpoints
mv cleargrasp-checkpoints cleargrasp-checkpoints.zip
wget http://clkgum.com/shreeyak/cleargrasp-dataset-test
mv cleargrasp-dataset-test cleargrasp-dataset-test.tar
wget http://clkgum.com/shreeyak/cleargrasp-dataset-train
mv cleargrasp-dataset-train cleargrasp-dataset-train.tar
unzip cleargrasp-checkpoints.zip
tar -xf cleargrasp-dataset-test-val.tar
tar -xf cleargrasp-dataset-train.tar
cd ..

