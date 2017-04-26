

#### Passing multiple channel sets: Align (Registration)
Typically the operation is only applied to single channel. For a jitter correction, you use single channel to calculate jitters and applied to multiple channels.  
To pass multiple channel sets, use " / " for separation of path after _-i_.

The following example uses CFP channel to calculate jitters and applied to CFP, TRITC and Far-red images. All the channels need to have the same number of images.

```
python celltk/preprocess.py -f align -i data/testimages4/imCFP* / data/testimages4/imTRITC* / data/testimages4/imFar-red* -p DOWN=2
```

#### Passing multiple parameters: multiple tracking
In track.py, at each frame, it attempts to link object from previous frame, typically using information about intensity and location.
If there are objects remained unlinked after the first attempt, they can be passed to the next tracking algorithm.  
In the following example, three methods are applied for tracking, which are "run_lap", "track_neck_cut" and another "track_neck_cut". To pass parameters for multiple functions, you need to separate them by " / ".  
e.g. the last "track_neck_cut" uses THRES_ANGLE=160.


```
python celltk/track.py -i d0/img_00000000* -l d1/img_00000000* -f run_lap track_neck_cut track_neck_cut -p DISPLACEMENT=50 MASSTHRES=0.25 / DISPLACEMENT=30 THRES_ANGLE=180 /  THRES_ANGLE=160
```

