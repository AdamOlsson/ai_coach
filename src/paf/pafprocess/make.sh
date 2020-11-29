echo "Changing directory for paf build."
cd paf/pafprocess

swig -python -c++ pafprocess.i
python3 setup.py build_ext --inplace

echo "Paf build done, returning to call directory."
cd ../..