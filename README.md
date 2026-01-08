# project-ai

export PYTHONPATH=$(pwd)

## Dataset

One can download the datasets (Squirrel, Chameleon, Actor) from the google drive link below:

https://drive.google.com/drive/folders/1rr3kewCBUvIuVxA6MJ90wzQuF-NnCRtf?usp=drive_link

For Chameleon and Squirrel, we use the [new splits](https://github.com/yandex-research/heterophilous-graphs/tree/main)that filter out the overlapped nodes.

Explain how to download datasets and where to place.

Splits:

for i in {0..9}; do
    wget "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/splits/film_split_0.6_0.2_${i}.npz" \
         -O "data/geom-gcn/film/film_split_0.6_0.2_${i}.npz"
done