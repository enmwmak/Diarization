#!/bin/bash -e
# Copyright 2017-2018  David Snyder
#           2017-2018  Matthew Maciejewski
#
# Apache 2.0.
#
# X-vector extractor with attention
# 
# Results on cuhkncd
# VAD     Part   Miss  FA   Spk-err   DER
#------------------------------------
# Aspire  0      10.8  1.0  3.7       15.50
# Aspire  1      10.6  0.9  4.6       16.04


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
vad=Aspire
p=1                                   # Part number
nnet_dir=exp/xvector_tf_attent

#======================================================
# Prepare datasets
#======================================================
if [ $stage -eq 0 ]; then
  rm -rf data/cuhkncd${p}  
  local/make_cuhkncd.pl /corpus/cuhkncd/Part${p} data/cuhkncd${p}

  # Define reference diarization labels
  cd data/cuhkncd$p
  ln -s /corpus/cuhkncd/Part$p/cuhkncd_ref.rttm fullref.rttm
  cd ../../  
fi

#========================================================================
# Compute MFCC and VAD for diarization dataset.
# Require GPU if Aspire VAD is used.
#========================================================================
if [ $stage -eq 1 ]; then

  nj=1    
  # Clean up segmented folder created by previous run of Stage 1
  rm -rf data/cuhkncd${p}_segmented* data/cuhkncd${p}_vad* tmp/sad_work
  rm -f mfcc/*cuhkncd${p}*
  mkdir -p tmp/sad_work  

  # Convert speech files to MFCC files (.ark and .scp) in $mfccdir  
  for name in cuhkncd${p}; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $nj \
      --cmd "$train_cmd" --write-utt2num-frames true \
      data/$name exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/$name
  done

  # Use either Kaldi VAD or Aspire VAD to determine the segment boundaries
  if [ $vad == "Kaldi" ]; then
      for name in cuhkncd${p}; do
	  sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
				      data/$name exp/make_vad $vaddir
	  utils/fix_data_dir.sh data/$name
      done
      # Convert data/cuhkncd${p} to data/cuhkncd${p}_segmented with segmentation info based on
      # the VAD obtained above
      diarization/vad_to_segments.sh data/cuhkncd${p} data/cuhkncd${p}_segmented
  elif [ $vad == "Aspire" ]; then
      local/detect_speech_activity.sh --nj $nj \
	  data/cuhkncd${p} exp/tdnn_stats_asr_sad_1a mfcc tmp/sad_work data/cuhkncd${p}_vad
      cd data; ln -s cuhkncd${p}_vad_seg cuhkncd${p}_segmented; cd ..
  else
      echo "vad should be either Kaldi or Aspire"
      exit 1
  fi
  
  # Convert speech files to MFCC files (.ark and .scp) in $mfccdir  
  for name in cuhkncd${p}_segmented; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $nj \
      --cmd "$train_cmd" --write-utt2num-frames true \
      data/$name exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/$name
  done

  # Unlike speaker verification, we pre-compute the CMN-MFCC frames and store them
  # in disk. Although this is somewhat wasteful in terms of disk space,
  # for diarization, it ends up being NOT preferable to performing CMN in memory.
  # Should CMN were performed in memory (e.g., we used --apply-cmn true in
  # diarization/nnet3/xvector/extract_xvectors.sh) it would need to be
  # performed after the subsegmentation, which leads to poorer results.
  # NB: Files in exp/${name}_cmn contains CMN-MFCCs. They will be used
  # by extract_xvectors.sh later. The script prepare_feats.sh will also
  # create the data in data/{sre,callhome1,callhome2}_cmn.
  for name in cuhkncd${p}_segmented; do
    local/nnet3/xvector/prepare_feats.sh --nj $nj --cmd "$train_cmd" \
      data/$name data/${name}_cmn exp/${name}_cmn
    #cp data/$name/vad.scp data/${name}_cmn/
    if [ -f data/$name/segments ]; then
      cp data/$name/segments data/${name}_cmn/
    fi
    utils/fix_data_dir.sh data/${name}_cmn
  done

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
  rm -rf $nnet_dir/xvectors_cuhkncd*
  gpu=true
  nj=1
  
  diarization/tf/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj $nj --window 1.5 --period 0.75 --apply-cmn false --use_gpu $gpu \
    --min-segment 0.5 $nnet_dir data/cuhkncd${p}_segmented_cmn $nnet_dir/xvectors_cuhkncd${p}_segmented

  # Bug fixed by M.W. Mak on 12 Aug. 19
  # Because segments shorter than sliding win size will be ignored, we need to fix the
  # "segments" file in $nnet_dir/xvectors_cuhkncd and 2. This will make this file contains
  # the same number of lines as $nnet_dir/xvectors_cuhkncd/xvectors.scp
  cp $nnet_dir/xvectors_cuhkncd${p}_segmented/xvector.scp $nnet_dir/xvectors_cuhkncd${p}_segmented/feats.scp
  cp data/cuhkncd${p}_segmented_cmn/wav.scp $nnet_dir/xvectors_cuhkncd${p}_segmented
  utils/fix_data_dir.sh $nnet_dir/xvectors_cuhkncd${p}_segmented
  
fi

#============================================================================================
# Train PLDA models
#============================================================================================
if [ $stage -eq 8 ]; then

  # Train a PLDA model using SRE data but use cuhkncd data for whitening
  # Output PLDA model to $nnet_dir/xvectors_cuhkncd${p}_segmented/plda 
  $train_cmd $nnet_dir/xvectors_cuhkncd${p}_segmented/log/plda.log \
    ivector-compute-plda ark:$nnet_dir/xvectors_sre_segmented_128k/spk2utt \
      "ark:ivector-subtract-global-mean \
      scp:$nnet_dir/xvectors_sre_segmented_128k/xvector.scp ark:- \
      | transform-vec $nnet_dir/xvectors_cuhkncd${p}_segmented/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
      $nnet_dir/xvectors_cuhkncd${p}_segmented/plda || exit 1;
  
  # Adapt the PLDA model using cuhkncd${p}_segmented xvectors
  $train_cmd $nnet_dir/xvectors_cuhkncd${p}_segmented/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
      $nnet_dir/xvectors_cuhkncd${p}_segmented/plda \
      "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_cuhkncd${p}_segmented/xvector.scp ark:- | \
         transform-vec $nnet_dir/xvectors_cuhkncd${p}_segmented/transform.mat ark:- ark:- | \
         ivector-normalize-length ark:- ark:- |" \
      $nnet_dir/xvectors_cuhkncd${p}_segmented/plda_adapt || exit 1;  

  # Replace un-adapted PLDA by adapted PLDA
  # Comment out the following 2 lines if unadapted PLDA model is used for scoring
  cp $nnet_dir/xvectors_cuhkncd${p}_segmented/plda_adapt $nnet_dir/xvectors_cuhkncd${p}_segmented/plda
  rm -f $nnet_dir/xvectors_cuhkncd${p}_segmented/plda_adapt
fi

#============================================================================================
# Perform PLDA scoring
#============================================================================================
if [ $stage -eq 9 ]; then

  # Bug fixed by M.W. Mak on 12 Aug. 19
  # To make sure that the segment file agrees with the utt2spk file, the plda_scores/tmp folder
  # needs the wav.scp (diarization/nnet3/xvector/score_plda.sh does not copy this file). This file
  # is required by utils/fix_data_dir.sh in score_plda.sh  
  mkdir -p $nnet_dir/xvectors_cuhkncd${p}_segmented/plda_scores/tmp
  cp data/cuhkncd${p}_segmented_cmn/wav.scp $nnet_dir/xvectors_cuhkncd${p}_segmented/plda_scores/tmp  
    
  # Perform PLDA scoring on all pairs of segments for each recording.
  diarization/tf/xvector/score_plda.sh --cmd "$train_cmd --mem 4G" \
    --nj 1 $nnet_dir/xvectors_cuhkncd${p}_segmented $nnet_dir/xvectors_cuhkncd${p}_segmented \
    $nnet_dir/xvectors_cuhkncd${p}_segmented/plda_scores

  # Bug fixed by M.W. Mak on 12 Aug. 19.
  # After computing the scores in plda_scores/, we need to run utils/fix_data_dir.sh on this folder
  cp data/cuhkncd${p}_segmented_cmn/wav.scp $nnet_dir/xvectors_cuhkncd${p}_segmented/plda_scores
  utils/fix_data_dir.sh $nnet_dir/xvectors_cuhkncd${p}_segmented/plda_scores
fi

#============================================================================================
# Cluster the PLDA scores using a stopping threshold.
#============================================================================================
if [ $stage -eq 10 ]; then
  echo "Do not optimize threshold"
fi

#============================================================================================
# Cluster the PLDA scores using the oracle number of speakers
# Diarization decisions stored in
# $nnet_dir/xvectors_cuhkncd${p}_segmented/plda_scores_num_spk/rttm
#============================================================================================
if [ $stage -eq 11 ]; then

  # Create reco2num_spk file containing only 2 speakers per utterance
  awk '{print $1 " 2"}' data/cuhkncd${p}/spk2utt > data/cuhkncd${p}/reco2num_spk   
    
  # In this section, we show how to do the clustering if the number of speakers
  # (and therefore, the number of clusters) per recording is known in advance.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" \
    --reco2num-spk data/cuhkncd${p}/reco2num_spk --nj 1 \
    $nnet_dir/xvectors_cuhkncd${p}_segmented/plda_scores \
    $nnet_dir/xvectors_cuhkncd${p}_segmented/plda_scores_num_spk

  # Compare the diarization decisions in $nnet_dir/xvectors_cuhkncd${p}_segmented/plda_scores_num_spk/rttm
  # with data/cuhkncd${p}/cuhkncd${p}_ref.rttm
  mkdir -p $nnet_dir/results
  cat $nnet_dir/xvectors_cuhkncd${p}_segmented/plda_scores_num_spk/rttm | \
      /usr/local/kaldi/tools/sctk-2.4.10/bin/md-eval.pl -1 -c 0.25 \
			       -r data/cuhkncd${p}/fullref.rttm -s - 2> \
			       $nnet_dir/results/num_spk.log \
			       > $nnet_dir/results/DER_num_spk.txt

  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' $nnet_dir/results/DER_num_spk.txt)
  echo "DER: $der%"
  miss=`cat $nnet_dir/results/DER_num_spk.txt | grep 'MISSED\ SPEAKER\ TIME' |  \
  	    				      cut -d'(' -f2 | awk '{print $1}'`
  echo "MISS: $miss%"
  fa=`cat $nnet_dir/results/DER_num_spk.txt | grep 'FALARM\ SPEAKER\ TIME' |  \
  	  				      cut -d'(' -f2 | awk '{print $1}'`
  echo "FA: $fa%"
  sr=`cat $nnet_dir/results/DER_num_spk.txt | grep 'SPEAKER\ ERROR\ TIME' |  \
  	  				      cut -d'(' -f2 | awk '{print $1}'`
  echo "Spk-err: $sr%"  

  
fi

#============================================================================================
# Find miss rate and false alarm of individual file
# This assumes that Step 11 has been run
#============================================================================================
if [ $stage -eq 12 ]; then

  # Get file list from ref rttm file
  temprttm=$nnet_dir/results/temp.rttm
  tempout=$nnet_dir/results/temp.out  
  filelist=`awk '{print $2}' data/cuhkncd${p}/fullref.rttm | uniq`
  for file in $filelist; do
      echo "$file:"
      cat data/cuhkncd${p}/fullref.rttm | grep $file > $temprttm
      cat $nnet_dir/xvectors_cuhkncd${p}_segmented/plda_scores_num_spk/rttm | grep $file | \
	  /usr/local/kaldi/tools/sctk-2.4.10/bin/md-eval.pl -1 -c 0.25 \
					-r $temprttm -s - 2> /dev/null > $tempout
      cat $tempout | grep 'MISSED SPEAKER TIME\|FALARM SPEAKER TIME'
  done
  rm -f $temprttm
  rm -f $tempout
fi

