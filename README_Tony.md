SparseTrack
--

0. Build docker image and continue from step 6
```sh
docker build -t <image_name> .
```
OR
```sh
docker pull 12labs/sparsetrack
```

1. Clone repo
```sh
git clone https://github.com/swmoon00/SparseTrack.git
```

2. Install apt packages (with sudo)
```sh
apt install libboost-all-dev
apt install libpython3-dev
```

3. Configure conda
   - The python version should be matched with `/usr/bin/python3 --version`
   e.g.
```sh
conda create -n sparsetrack python=3.8.10
```

4. Install pbcvt package and copy library file
   Edit `CMakeLists.txt` file as in SparseTrack instructions, and compile with `make`

5. Install cython_bbox
   - There is a bug in cython_bbox package.
   - Install patched version
```sh
pip install git+https://github.com/swmoon00/cython_bbox.git
```

6. Download model weights
   - https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view
   - Change `train.init_checkpoint` option in `mot17_track_cfg.py` file

7. Run demo
```sh
python demo_track.py --config-file mot17_track_cfg.py --video-input videos/palace.mp4
```