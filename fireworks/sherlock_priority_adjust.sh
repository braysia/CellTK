#!/bin/bash
while [ "1" = "1" ]; do
    echo $(date +"%Y.%m.%d %r") "Checking for conflicting jobs..."

    #check for pending jobs in mcovert queue with command ‘bash'
    if [[ $(squeue -p mcovert -h -t PD -n bash) ]]; then
        echo "Conflicting job found."
        echo "Checking for jobs to hold..."

        #loop over all jobs that USER has pending in mcovert queue
        for i in $(squeue -p mcovert -u $USER -h -t PD -o %i)
        do
            #exclude USER jobs that are also command ‘bash'
            if [ $(squeue -j $i -h -t PD -o %o) != "bash" ]; then
                #hold all pending jobs that are not command ‘bash'
                scontrol hold jobid=$i
                echo "Held job" $i
            else
                echo "No change to job" $i
            fi
        done
    else
        echo "No conflicting jobs found."
        echo "Checking for jobs to release..."
    
        #loop over all jobs that USER has pending in mcovert queue
        for i in $(squeue -p mcovert -u $USER -h -t PD -o %i)
        do
            #if job is held by USER, release it
            if [ $(squeue -j $i -h -t PD -o %r) == "JobHeldUser" ]; then
                scontrol release jobid=$i
                echo "Released job" $i
            fi
        done
    fi
    echo "Complete. Waiting 3 minutes..."
    sleep 180
done
