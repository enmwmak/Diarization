#!/bin/bash -e
# Copyright 2017-2018  David Snyder
#           2017-2018  Matthew Maciejewski
#
# Apache 2.0.
#
# This recipe demonstrates the use of BUT x-vectors for speaker diarization.
# The scripts are based on the recipe in ../v2-1/run.sh.
# It is similar to the x-vector-based diarization system
# described in "Diarization is Hard: Some Experiences and Lessons Learned for
# the JHU Team in the Inaugural DIHARD Challenge" by Sell et al.  The main
# difference is that we haven't implemented the VB resegmentation yet.

# Modified by M.W. Mak in Aug. 2019 for enmcompX systems.
# Performance in DER of BUT x-vector extractor
#                              w/o PLDA Adapt    w/ PLDA Adapt
#     No. of speakers unknown: 8.29%             8.69%   
#     No. of speakers known    7.17%             7.15%
# Performance in DER of Kaldi baseline
#     No. of speakers unknown: 8.39%
#     No. of speakers known    7.12%


# Get run level
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <stage>"
fi
stage=$1

#======================================================
# Set up environment
#======================================================
. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
nnet_dir=exp/xvector_tf_attent
#nnet_dir=exp/tmp
#nnet_dir=exp/xvector_tf_attent_sm     # Same size as the Kaldi DNN in v2-1

#======================================================
# Prepare datasets
#======================================================
if [ $stage -eq 0 ]; then
  echo "Run the run.sh under v2-1 to prepare the data folder"
fi

#========================================================================
# Prepare features (compute MFCCs). No GPU required. Only works on
# enmcomp2,3,6,12 because it requires the version of compute-mfcc-feats with
# the option --write-utt2dur available.
#========================================================================
if [ $stage -eq 1 ]; then
  echo "Run the run.sh under v2-1 to get the MFCC and VAD"
fi

#============================================================================================
# Augment the training data with reverberation, noise, music, and babble, and combined
# them with the clean data. The combined list in data/train_combined (comprises clean and
# augmented SWBD+SRE) will be used to train the xvector DNN. The SRE subset will be used
# to train the PLDA model. 
# Required Anaconda python 3.7 (conda activate tfenv)
#============================================================================================
if [ $stage -eq 2 ]; then
  echo "Run the run.sh in v2-1 to get the augmented data"

  # Because v2-2/ has its own exp/ folder, we need to run the following again
  # Make MFCC for the augmented data. Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 16 --cmd "$train_cmd" \
    data/train_aug_128k exp/make_mfcc $mfccdir

fi

#============================================================================================
# Now we prepare the features to generate examples for xvector training.
# This stage produces {data,exp}/train_combined_cmn_no_sil
#============================================================================================
if [ $stage -eq 3 ]; then
   echo "Run the run.sh in v2-1/ and create symbolic link to exp/train_combined_cmn_no_sil"
   rm -f exp/train_combined_cmn_no_sil
   cd exp
   ln -s ../../v2-1/exp/train_combined_cmn_no_sil .
   cd ..  
fi

#============================================================================================
# Execute local/tf/get_egs.sh to generate archives 
# stage=6 in run_xvector.sh will execute local/tf/get_egs.sh to produce archive
# files in .npy and .tar format in ${nnet_dir}/egs 
# Do not required GPU. Run under Anaconda3 py27 env
#============================================================================================
if [ ${stage} -eq 4 ]; then  
  local/tf/run_xvector.sh --stage 6 --train-stage -1 \
			  --data data/train_combined_cmn_no_sil --nnet-dir ${nnet_dir} \
			  --egs-dir ${nnet_dir}/egs

fi

#============================================================================================
# Train x-vector extractor
# stage=7 in run_xvector.sh will execute local/tf/train_dnn.py to train
# x-vector extractor
# Require GPU. Only work on Tensorflow 1.x and Python2.7. Work only under Anaconda3
# py27 or py27-tf1.14 env in enmcomp3 because of train_dnn.py does not work on Tensorflow 2.0
# --train-stage=-1 means starting from randomly initialized network.
# --train-stage=256 means starting from previous network stored in folder model_256/.
# To increase the no. of epochs, change the parameter --num-epochs=6 when calling
# local/tf/train_dnn.py in local/tf/run_xvector.sh
#============================================================================================
if [ ${stage} -eq 5 ]; then  
  local/tf/run_xvector.sh --stage 7 --train-stage -1 \
			  --data data/train_combined_cmn_no_sil --nnet-dir ${nnet_dir} \
			  --egs-dir ${nnet_dir}/egs
fi

#============================================================================================
# Extract x-vectors. No. of jobs (--nj) can only be one; otherwise not enough GPU memory
# Require GPU. Run under Anaconda env py27-tf1.14.
# No. of jobs (--nj) is 1 if there is only 1 GPU
# No. of jobs (--nj) is 8 if CPU is used (gpu=false)
# extract_xvectors.sh also compute mean.vec and PCA transform.mat to the output dir
#============================================================================================
if [ $stage -eq 7 ]; then
    
  # Clean up x-vector folders to avoid errors in extract_xvectors.sh
  #rm -rf $nnet_dir/xvectors_*
  gpu=true
  nj=1
 
  # Extract x-vectors for the two partitions of callhome. Note that segments
  # less than sliding window size (--window) will be ignored.
  # Use sliding winsize of 1.5s, shifted by 0.75s. Min segment-length per x-vector is 0.5s.
  # This script will create subsegments of size 1.5s.
  # No need to apply CMN as the data in data/callhome1_cmn have been CMN-ed already.   
  diarization/tf/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj $nj --window 1.5 --period 0.75 --apply-cmn false --use_gpu $gpu \
    --min-segment 0.5 $nnet_dir data/callhome1_cmn $nnet_dir/xvectors_callhome1

  diarization/tf/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj $nj --window 1.5 --period 0.75 --apply-cmn false --use_gpu $gpu \
    --min-segment 0.5 $nnet_dir data/callhome2_cmn $nnet_dir/xvectors_callhome2

  # Reduce the amount of training data for the PLDA,
  utils/subset_data_dir.sh data/sre_cmn_segmented 128000 data/sre_cmn_segmented_128k
  # Extract x-vectors for the SRE, which is our PLDA training
  # data.  A long period is used here so that we don't compute too
  # many x-vectors for each recording.
  diarization/tf/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj $nj --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false --use_gpu $gpu \
    --hard-min true $nnet_dir data/sre_cmn_segmented_128k $nnet_dir/xvectors_sre_segmented_128k

  # Extract x-vectors for PLDA adaptation
  diarization/tf/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj $nj --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false --use_gpu true \
    --hard-min true $nnet_dir data/callhome1_cmn $nnet_dir/xvectors_callhome1_plda

  diarization/tf/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj $nj --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false --use_gpu true \
    --hard-min true $nnet_dir data/callhome2_cmn $nnet_dir/xvectors_callhome2_plda

  # Bug fixed by M.W. Mak on 12 Aug. 19
  # Because segments shorter than sliding win size will be ignored, we need to fix the
  # "segments" file in $nnet_dir/xvectors_callhome1 and 2. This will make this file contains
  # the same number of lines as $nnet_dir/xvectors_callhome1/xvectors.scp
  cp $nnet_dir/xvectors_callhome1/xvector.scp $nnet_dir/xvectors_callhome1/feats.scp
  cp $nnet_dir/xvectors_callhome2/xvector.scp $nnet_dir/xvectors_callhome2/feats.scp
  cp data/callhome1_cmn/wav.scp $nnet_dir/xvectors_callhome1
  cp data/callhome2_cmn/wav.scp $nnet_dir/xvectors_callhome2
  utils/fix_data_dir.sh $nnet_dir/xvectors_callhome1
  utils/fix_data_dir.sh $nnet_dir/xvectors_callhome2
  
fi

#============================================================================================
# Train PLDA models
#============================================================================================
if [ $stage -eq 8 ]; then

  # Train a PLDA model on SRE, using callhome1 to whiten.
  # We will later use this to score x-vectors in callhome2.
  $train_cmd $nnet_dir/xvectors_callhome1/log/plda.log \
    ivector-compute-plda ark:$nnet_dir/xvectors_sre_segmented_128k/spk2utt \
      "ark:ivector-subtract-global-mean \
      scp:$nnet_dir/xvectors_sre_segmented_128k/xvector.scp ark:- \
      | transform-vec $nnet_dir/xvectors_callhome1/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
      $nnet_dir/xvectors_callhome1/plda || exit 1;

  # Adapt the PLDA model
  $train_cmd $nnet_dir/xvectors_callhome1/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
      $nnet_dir/xvectors_callhome1/plda \
      "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_callhome1_plda/xvector.scp ark:- | \
         transform-vec $nnet_dir/xvectors_callhome1/transform.mat ark:- ark:- | \
         ivector-normalize-length ark:- ark:- |" \
      $nnet_dir/xvectors_callhome1/plda_adapt || exit 1;  

  # Train a PLDA model on SRE, using callhome2 to whiten.
  # We will later use this to score x-vectors in callhome1.
  $train_cmd $nnet_dir/xvectors_callhome2/log/plda.log \
    ivector-compute-plda ark:$nnet_dir/xvectors_sre_segmented_128k/spk2utt \
      "ark:ivector-subtract-global-mean \
      scp:$nnet_dir/xvectors_sre_segmented_128k/xvector.scp ark:- \
      | transform-vec $nnet_dir/xvectors_callhome2/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
      $nnet_dir/xvectors_callhome2/plda || exit 1;

  # Adapt the PLDA model
  $train_cmd $nnet_dir/xvectors_callhome2/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
      $nnet_dir/xvectors_callhome2/plda \
      "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_callhome2_plda/xvector.scp ark:- | \
         transform-vec $nnet_dir/xvectors_callhome2/transform.mat ark:- ark:- | \
         ivector-normalize-length ark:- ark:- |" \
      $nnet_dir/xvectors_callhome2/plda_adapt || exit 1;  

  # Replace un-adapted PLDA by adapted PLDA
  # Comment out the following 4 lines if unadapted PLDA model is used for scoring
  #cp $nnet_dir/xvectors_callhome1/plda_adapt $nnet_dir/xvectors_callhome1/plda
  #cp $nnet_dir/xvectors_callhome2/plda_adapt $nnet_dir/xvectors_callhome2/plda
  #rm -f $nnet_dir/xvectors_callhome1/plda_adapt
  #rm -f $nnet_dir/xvectors_callhome2/plda_adapt

fi

#============================================================================================
# Perform PLDA scoring
#============================================================================================
if [ $stage -eq 9 ]; then

  # Bug fixed by M.W. Mak on 12 Aug. 19
  # To make sure that the segment file agrees with the utt2spk file, the plda_scores/tmp folder
  # needs the wav.scp (diarization/nnet3/xvector/score_plda.sh does not copy this file). This file
  # is required by utils/fix_data_dir.sh in score_plda.sh  
  mkdir -p $nnet_dir/xvectors_callhome1/plda_scores/tmp
  mkdir -p $nnet_dir/xvectors_callhome2/plda_scores/tmp  
  cp data/callhome1_cmn/wav.scp $nnet_dir/xvectors_callhome1/plda_scores/tmp  
  cp data/callhome2_cmn/wav.scp $nnet_dir/xvectors_callhome2/plda_scores/tmp  
    
  # Perform PLDA scoring on all pairs of segments for each recording.
  # The first directory contains the PLDA model that used callhome2
  # to perform whitening (recall that we're treating callhome2 as a
  # held-out dataset).  The second directory contains the x-vectors
  # for callhome1.
  diarization/tf/xvector/score_plda.sh --cmd "$train_cmd --mem 4G" \
    --nj 8 $nnet_dir/xvectors_callhome2 $nnet_dir/xvectors_callhome1 \
    $nnet_dir/xvectors_callhome1/plda_scores

  # Do the same thing for callhome2.
  diarization/tf/xvector/score_plda.sh --cmd "$train_cmd --mem 4G" \
    --nj 8 $nnet_dir/xvectors_callhome1 $nnet_dir/xvectors_callhome2 \
    $nnet_dir/xvectors_callhome2/plda_scores

  # Bug fixed by M.W. Mak on 12 Aug. 19.
  # After computing the scores in plda_scores/, we need to run utils/fix_data_dir.sh on this folder
  cp data/callhome1_cmn/wav.scp $nnet_dir/xvectors_callhome1/plda_scores
  cp data/callhome2_cmn/wav.scp $nnet_dir/xvectors_callhome2/plda_scores
  utils/fix_data_dir.sh $nnet_dir/xvectors_callhome1/plda_scores
  utils/fix_data_dir.sh $nnet_dir/xvectors_callhome2/plda_scores
fi

#============================================================================================
# Cluster the PLDA scores using a stopping threshold.
#============================================================================================
if [ $stage -eq 10 ]; then
  # First, we find the threshold that minimizes the DER on each partition of callhome.
  mkdir -p $nnet_dir/tuning
  for dataset in callhome1 callhome2; do
    echo "Tuning clustering threshold for $dataset"
    best_der=100
    best_threshold=0
    utils/filter_scp.pl -f 2 data/$dataset/wav.scp \
      data/callhome/fullref.rttm > data/$dataset/ref.rttm

    # The threshold is in terms of the log likelihood ratio provided by the
    # PLDA scores.  In a perfectly calibrated system, the threshold is 0.
    # In the following loop, we evaluate the clustering on a heldout dataset
    # (callhome1 is heldout for callhome2 and vice-versa) using some reasonable
    # thresholds for a well-calibrated system.
    for threshold in -0.3 -0.2 -0.1 -0.05 0 0.05 0.1 0.2 0.3; do
      diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 1 \
        --threshold $threshold $nnet_dir/xvectors_$dataset/plda_scores \
        $nnet_dir/xvectors_$dataset/plda_scores_t$threshold

      echo "Running md-eval.pl"
      /usr/local/kaldi/tools/sctk-2.4.10/bin/md-eval.pl -1 -c 0.25 -r data/$dataset/ref.rttm \
				    -s $nnet_dir/xvectors_$dataset/plda_scores_t$threshold/rttm \
				    2> $nnet_dir/tuning/${dataset}_t${threshold}.log \
				    > $nnet_dir/tuning/${dataset}_t${threshold}

      der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
        $nnet_dir/tuning/${dataset}_t${threshold})
      if [ $(echo $der'<'$best_der | bc -l) -eq 1 ]; then
        best_der=$der
        best_threshold=$threshold
      fi
    done
    echo "$best_threshold" > $nnet_dir/tuning/${dataset}_best
  done

  # Cluster callhome1 using the best threshold found for callhome2.  This way,
  # callhome2 is treated as a held-out dataset to discover a reasonable
  # stopping threshold for callhome1.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 8 \
    --threshold $(cat $nnet_dir/tuning/callhome2_best) \
    $nnet_dir/xvectors_callhome1/plda_scores $nnet_dir/xvectors_callhome1/plda_scores

  # Do the same thing for callhome2, treating callhome1 as a held-out dataset
  # to discover a stopping threshold.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 8 \
    --threshold $(cat $nnet_dir/tuning/callhome1_best) \
    $nnet_dir/xvectors_callhome2/plda_scores $nnet_dir/xvectors_callhome2/plda_scores

  mkdir -p $nnet_dir/results
  # Now combine the results for callhome1 and callhome2 and evaluate it
  # together.
  cat $nnet_dir/xvectors_callhome1/plda_scores/rttm \
      $nnet_dir/xvectors_callhome2/plda_scores/rttm | \
      /usr/local/kaldi/tools/sctk-2.4.10/bin/md-eval.pl -1 -c 0.25 -r \
    data/callhome/fullref.rttm -s - 2> $nnet_dir/results/threshold.log \
    > $nnet_dir/results/DER_threshold.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $nnet_dir/results/DER_threshold.txt)
  # Using supervised calibration, DER: 8.39%
  # Compare to 10.36% in ../v1/run.sh
  echo "Using supervised calibration, DER: $der%"
fi


#============================================================================================
# Cluster the PLDA scores using the oracle number of speakers
#============================================================================================
if [ $stage -eq 11 ]; then

  # In this section, we show how to do the clustering if the number of speakers
  # (and therefore, the number of clusters) per recording is known in advance.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" \
    --reco2num-spk data/callhome1/reco2num_spk \
    $nnet_dir/xvectors_callhome1/plda_scores $nnet_dir/xvectors_callhome1/plda_scores_num_spk

  diarization/cluster.sh --cmd "$train_cmd --mem 4G" \
    --reco2num-spk data/callhome2/reco2num_spk \
    $nnet_dir/xvectors_callhome2/plda_scores $nnet_dir/xvectors_callhome2/plda_scores_num_spk

  mkdir -p $nnet_dir/results
  # Now combine the results for callhome1 and callhome2 and evaluate it together.
  cat $nnet_dir/xvectors_callhome1/plda_scores_num_spk/rttm \
  $nnet_dir/xvectors_callhome2/plda_scores_num_spk/rttm \
      | /usr/local/kaldi/tools/sctk-2.4.10/bin/md-eval.pl -1 -c 0.25 \
			       -r data/callhome/fullref.rttm -s - 2> \
			       $nnet_dir/results/num_spk.log \
			       > $nnet_dir/results/DER_num_spk.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $nnet_dir/results/DER_num_spk.txt)
  # Using the oracle number of speakers, DER: 7.12%
  # Compare to 8.69% in ../v1/run.sh
  echo "Using the oracle number of speakers, DER: $der%"
fi
