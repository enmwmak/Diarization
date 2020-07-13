#!/bin/bash -e
# Copyright 2017-2018  David Snyder
#           2017-2018  Matthew Maciejewski
#
# Apache 2.0.
#
# This recipe demonstrates the use of x-vectors for speaker diarization.
# The scripts are based on the recipe in ../v1/run.sh, but clusters x-vectors
# instead of i-vectors.  It is similar to the x-vector-based diarization system
# described in "Diarization is Hard: Some Experiences and Lessons Learned for
# the JHU Team in the Inaugural DIHARD Challenge" by Sell et al.  The main
# difference is that we haven't implemented the VB resegmentation yet.

# Modified by M.W. Mak in Aug. 2019 for enmcompX systems.
# Performance in DER
#                              w/o PLDA Adapt    w/ PLDA Adapt
#     No. of speakers unknown: 7.32%             7.64%
#     No. of speakers known    6.29%             6.80%
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
nnet_dir=exp/xvector_nnet_2a/
#nnet_dir=exp/xvector_nnet_tmp/       # For debugging

#======================================================
# Prepare datasets
#======================================================
if [ $stage -eq 0 ]; then

  # Prepare a collection of NIST SRE data. This will be used to train
  # x-vector DNN and PLDA model. This script will create data/sre, which
  # comprises sre04-08 data with speaker labels but no segmentation info
  local/make_sre.sh /corpus data

  # Prepare SWB for x-vector DNN training. For symbolic link dir, we need
  # to append '/' to the end of the dir name
  local/make_swbd2_phase1.pl /corpus/swb2ph1/ data/swbd2_phase1
  local/make_swbd2_phase2.pl /corpus/swb2ph2/ data/swbd2_phase2
  local/make_swbd2_phase3.pl /corpus/swb2ph3/ data/swbd2_phase3
  local/make_swbd_cellular1.pl /corpus/swbcell1/ data/swbd_cellular1
  local/make_swbd_cellular2.pl /corpus/swbcell2/ data/swbd_cellular2

  # Prepare the Callhome portion of NIST SRE 2000. Split the 500 speakers into
  # data/callhome1 and data/callhome2 with 250 speakers each.
  # This script will create data/callhome1 and data/callhome2 with ground-truth
  # diarization labels (in files data/callhome/fullref.rttm).
  # Data in callhome1 and callhome2 are treated like a held-out
  # dataset, and will be used to estimate various quantities needed to perform
  # diarization on the other part (and vice versa).
  local/make_callhome.sh /corpus/nist00/ data

  # Combine swbd2_ph1-3, swbd_cellular1-2 and sre datasets into data/train
  # The data/train and their augmentation will be used for training x-vector DNN
  utils/combine_data.sh data/train data/swbd_cellular1 data/swbd_cellular2 \
    data/swbd2_phase1 data/swbd2_phase2 data/swbd2_phase3 data/sre

fi

#========================================================================
# Prepare features (compute MFCCs). No GPU required. 
# Requires the version of compute-mfcc-feats with the option --write-utt2dur
#========================================================================
if [ $stage -eq 1 ]; then

  # Convert speech files to MFCC files (.ark and .scp) in $mfccdir  
  for name in train callhome1 callhome2; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 16 \
      --cmd "$train_cmd" --write-utt2num-frames true \
      data/$name exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/$name
  done
  
  # Determine speech/nonspeech segments and store the segment files
  # (.ark and .scp) in $vaddir 
  for name in train callhome1 callhome2; do
    sid/compute_vad_decision.sh --nj 16 --cmd "$train_cmd" \
      data/$name exp/make_vad $vaddir
    utils/fix_data_dir.sh data/$name
  done

  # Because data/sre dataset is a subset of data/train, we do not have to
  # compute the MFCC and VAD for data/sre. We just copy data/train/feats.scp
  # and data/train/vad.scp to data/sre and filter out the non-sre lines in
  # these two files based on the info in data/sre/utt2spk and data/sre/spk2utt
  cp data/train/{feats,vad}.scp data/sre/
  utils/fix_data_dir.sh data/sre

  # Unlike speaker verification, we pre-compute the CMN-MFCC frames and store them
  # in disk. Although this is somewhat wasteful in terms of disk space,
  # for diarization, it ends up being NOT preferable to performing CMN in memory.
  # Should CMN were performed in memory (e.g., we used --apply-cmn true in
  # diarization/nnet3/xvector/extract_xvectors.sh) it would need to be
  # performed after the subsegmentation, which leads to poorer results.
  # NB: Files in exp/${name}_cmn contains CMN-MFCCs. They will be used
  # by extract_xvectors.sh later. The script prepare_feats.sh will also
  # create the data in data/{sre,callhome1,callhome2}_cmn.
  for name in sre callhome1 callhome2; do
    local/nnet3/xvector/prepare_feats.sh --nj 16 --cmd "$train_cmd" \
      data/$name data/${name}_cmn exp/${name}_cmn
    cp data/$name/vad.scp data/${name}_cmn/
    if [ -f data/$name/segments ]; then
      cp data/$name/segments data/${name}_cmn/
    fi
    utils/fix_data_dir.sh data/${name}_cmn
  done

  echo "0.01" > data/sre_cmn/frame_shift
  # Create segments that will be used for extracting x-vectors for
  # training PLDA models. The segments are created using an energy-based VAD.
  # Exclude subsegments with duration < 0.25s. Store segment info in
  # the file data/sre_cmn_segmented/segments.
  diarization/vad_to_segments.sh --nj 16 --cmd "$train_cmd" \
    data/sre_cmn data/sre_cmn_segmented
fi

#============================================================================================
# Augment the training data with reverberation, noise, music, and babble, and combined
# them with the clean data. The combined list in data/train_combined (comprises clean and
# augmented SWBD+SRE) will be used to train the xvector DNN. The SRE subset will be used
# to train the PLDA model.
# Required Anaconda python 3.7 (conda activate tfenv)
#============================================================================================
if [ $stage -eq 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train/utt2num_frames > data/train/reco2dur
  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
  # additive noise here. We use the VAD from the original clean speech for the
  # reverberated speech
  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 8000 \
    data/train data/train_reverb
  cp data/train/vad.scp data/train_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/train_reverb data/train_reverb.new
  rm -rf data/train_reverb
  mv data/train_reverb.new data/train_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  local/make_musan.sh /corpus/musan data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  python steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train data/train_noise
  # Augment with musan_music
  python steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train data/train_music
  # Augment with musan_speech
  python steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train data/train_babble

  # Combine reverb, noise, music, and babble into one directory, data/train_aug
  utils/combine_data.sh data/train_aug data/train_reverb data/train_noise data/train_music data/train_babble

  # Take a random subset of the augmentations (128k is somewhat larger than twice
  # the size of the SWBD+SRE list)
  utils/subset_data_dir.sh data/train_aug 128000 data/train_aug_128k
  utils/fix_data_dir.sh data/train_aug_128k

  # Make MFCC for the augmented data. Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 16 --cmd "$train_cmd" \
    data/train_aug_128k exp/make_mfcc $mfccdir

  # Combine the clean and augmented SWBD+SRE list into data/train_combined.
  # This is now roughly double the size of the original clean list.
  utils/combine_data.sh data/train_combined data/train_aug_128k data/train
fi

#============================================================================================
# Now we prepare the features to generate examples for training x-vector extractor
# This stage produces {data,exp}/train_combined_cmn_no_sil
#============================================================================================
if [ $stage -eq 3 ]; then
  # This script applies CMN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 16 --cmd "$train_cmd" \
    data/train_combined data/train_combined_cmn_no_sil exp/train_combined_cmn_no_sil
  utils/fix_data_dir.sh data/train_combined_cmn_no_sil

  # Now, we need to remove features that are too short after removing silence
  # frames.  We want at least 5s (500 frames) per utterance.
  min_len=500
  mv data/train_combined_cmn_no_sil/utt2num_frames data/train_combined_cmn_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/train_combined_cmn_no_sil/utt2num_frames.bak > data/train_combined_cmn_no_sil/utt2num_frames
  utils/filter_scp.pl data/train_combined_cmn_no_sil/utt2num_frames data/train_combined_cmn_no_sil/utt2spk > data/train_combined_cmn_no_sil/utt2spk.new
  mv data/train_combined_cmn_no_sil/utt2spk.new data/train_combined_cmn_no_sil/utt2spk
  utils/fix_data_dir.sh data/train_combined_cmn_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/train_combined_cmn_no_sil/spk2utt > data/train_combined_cmn_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' \
    data/train_combined_cmn_no_sil/spk2num | utils/filter_scp.pl - data/train_combined_cmn_no_sil/spk2utt \
    > data/train_combined_cmn_no_sil/spk2utt.new
  mv data/train_combined_cmn_no_sil/spk2utt.new data/train_combined_cmn_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/train_combined_cmn_no_sil/spk2utt > data/train_combined_cmn_no_sil/utt2spk

  utils/filter_scp.pl data/train_combined_cmn_no_sil/utt2spk data/train_combined_cmn_no_sil/utt2num_frames > data/train_combined_cmn_no_sil/utt2num_frames.new
  mv data/train_combined_cmn_no_sil/utt2num_frames.new data/train_combined_cmn_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/train_combined_cmn_no_sil
fi

#============================================================================================
# Prepare archives in $nnet_dir/egs for training x-vector extractor using data specified in
# data/train_combined_cmn_no_sil. with 4979 speakers and 184533 utterances.
# Best run in enmcomp12 with Anaconda3 py27-tf1.14
#============================================================================================
if [ $stage -eq 4 ]; then
    local/nnet3/xvector/tuning/run_xvector_1a.sh --stage 4 --train-stage -1 \
						 --data data/train_combined_cmn_no_sil \
						 --nnet-dir $nnet_dir \
						 --egs-dir $nnet_dir/egs
fi

#============================================================================================
# Train x-vector extractor. Require GPU. Run under Anaconda env py27-tf1.14
# The final DNN will be stored in $nnet_dir/final.raw
#============================================================================================
if [ $stage -eq 5 ]; then
    local/nnet3/xvector/tuning/run_xvector_1a.sh --stage 5 --train-stage -1 \
						 --data data/train_combined_cmn_no_sil \
						 --nnet-dir $nnet_dir \
						 --egs-dir $nnet_dir/egs
fi
    
#============================================================================================
# Extract x-vectors. Require GPU. Run under Anaconda env py27-tf1.14.
# No. of jobs (--nj) is 1 if there is only 1 GPU
# No. of jobs (--nj) is 8 if CPU is used (gpu=false)
#============================================================================================
if [ $stage -eq 7 ]; then
    
  # Clean up x-vector folders to avoid errors in extract_xvectors.sh
  rm -rf $nnet_dir/xvectors_*
  gpu=false  
  nj=8
  
  # Extract x-vectors for the two partitions of callhome. Note that segments
  # less than sliding window size (--window) will be ignored.
  # Use sliding winsize of 1.5s, shifted by 0.75s. Min segment-length per x-vector is 0.5s.
  # This script will create subsegments of size 1.5s.
  # No need to apply CMN as the data in data/callhome1_cmn have been CMN-ed already.   
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj $nj --window 1.5 --period 0.75 --apply-cmn false --use_gpu $gpu \
    --min-segment 0.5 $nnet_dir data/callhome1_cmn $nnet_dir/xvectors_callhome1
  
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj $nj --window 1.5 --period 0.75 --apply-cmn false --use_gpu $gpu \
    --min-segment 0.5 $nnet_dir data/callhome2_cmn $nnet_dir/xvectors_callhome2

  # Reduce the amount of training data for the PLDA,
  utils/subset_data_dir.sh data/sre_cmn_segmented 128000 data/sre_cmn_segmented_128k
  # Extract x-vectors for the SRE, which is our PLDA training
  # data.  A long period is used here so that we don't compute too
  # many x-vectors for each recording.
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" \
    --nj $nj --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false --use_gpu $gpu \
    --hard-min true $nnet_dir data/sre_cmn_segmented_128k $nnet_dir/xvectors_sre_segmented_128k

  # Extract x-vectors for PLDA adaptation
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" \
    --nj $nj --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false --use_gpu $gpu \
    --hard-min true $nnet_dir data/callhome1_cmn $nnet_dir/xvectors_callhome1_plda
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" \
    --nj $nj --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false --use_gpu $gpu \
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

  # Train a PLDA model on SRE, using callhome1 to whiten (transform.mat).
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
  diarization/nnet3/xvector/score_plda.sh --cmd "$train_cmd --mem 4G" \
    --nj 8 $nnet_dir/xvectors_callhome2 $nnet_dir/xvectors_callhome1 \
    $nnet_dir/xvectors_callhome1/plda_scores

  # Do the same thing for callhome2.
  diarization/nnet3/xvector/score_plda.sh --cmd "$train_cmd --mem 4G" \
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
