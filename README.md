# CellTK


```
csegment -i c0/img_00* -f constant_thres -p THRES=2000 -o c1
ctrack -i c0/img_00* -l c1/img_00* -f run_lap track_neck_cut -o c1
cpostprocess -i c0/img_00* -l c1/img_00* -f gap_closing -o nuc
csubdetect -i nuc/img_00* -l -f ring_dilation -o cyto
capply -i c0/img_00* -l nuc/img_00* -o pos1
capply -i c0/img_00* -l cyto/img_00* -o pos1
```