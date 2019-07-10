rm *.log
rm *.json
rm *.nvvp
rm *.pth

PREFIX="python3.6 -u "
SUFFIX=
single_gpu=1
nvprof_on=0
multi=0
dist=0

for var in "$@"
do
    if [ $var = "cupti" ]; then
        SUFFIX="${SUFFIX} --cupti"
    fi
    if [ $var = "multi" ]; then # run multi-GPU profile
        single_gpu=0
        multi=1
        PREFIX="${PREFIX}-m multiproc "
    fi
    if [ $var = "nvprof" ]; then # use nvprof, cannot coexist with cupti
        nvprof_on=1
    fi
    if [ $var = "profile" ]; then # turn on NeuralTap profile
        SUFFIX="${SUFFIX} --profile"
    fi
    if [ $var = "dist" ]; then
        single_gpu=0
        dist=1
        PREFIX="${PREFIX}-m dist_multiproc "
        SUFFIX="${SUFFIX} --dist-url env://"
    fi
done

if [ $nvprof_on = 1 ] && [ $single_gpu = 0 ]; then
    if [ $multi = 1 ]; then
        PREFIX="nvprof --export-profile resnet50_multi_%p.nvvp -f ${PREFIX}"
    fi
    if [ $dist = 1 ]; then
        PREFIX="nvprof --export-profile resnet50_dist_%p.nvvp -f ${PREFIX}"
    fi
elif [ $nvprof_on = 1 ]; then
    PREFIX="nvprof --export-profile resnet50.nvvp -f ${PREFIX}"
fi

# run training
if [ $single_gpu = 1 ]; then
    echo "using single GPU"
    export CUDA_VISIBLE_DEVICES=0
fi

$PREFIX train.py --model resnet50 -b 32 -j 4 --data-path /mnt/dataset/imagenet $SUFFIX

