#!/bin/bash

while [ "1" = "1" ]; do
		echo $(date +"%Y.%m.%d.%S.%N") lpad detect_lostruns --rerun
        lpad detect_lostruns --rerun
        echo "Waiting 3 min"
        sleep 180
done
