nvcc merged.cu -o train -I/usr/local/HDF_Group/HDF5/1.14.5/include -L/usr/local/HDF_Group/HDF5/1.14.5/lib -lhdf5 -lhdf5_cpp 

export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/HDF_Group/HDF5/1.14.5/lib