#!/bin/bash

for scene in frl_apartment_2 \
            frl_apartment_3 frl_apartment_4 frl_apartment_5 office_0 office_4 room_0; do
    bash ./run_pipeline.sh $scene
done
