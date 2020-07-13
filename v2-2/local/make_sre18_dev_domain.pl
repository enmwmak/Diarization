#!/usr/bin/perl
use warnings; #sed replacement for -w perl parameter
# Copyright 2017   David Snyder
# Apache 2.0
#
# Make file for creating data/sre18_dev_cmn2 and data/sre18_dev_vast

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-SRE18-dev> <path-to-output>\n";
  print STDERR "e.g. $0 /corpus/sre18-dev data/\n";
  exit(1);
}

($db_base, $out_dir) = @ARGV;

#================================================================================
# Read /corpus/sre18-dev/docs/sre18_dev_trial_key.tsv to create utt2subsource map
#================================================================================
open(KEY, "</corpus/sre18-dev/docs/sre18_dev_trial_key.tsv") || die "Fail to open key file";
<KEY>;          #Skip header
my %utt2subsource = ();
while (my $line=<KEY>) {
    chomp($line);
    @t = split('\s+', $line);
    $subsource = $t[7];
    $utt = $t[1];
    $utt2subsource{$utt} = $subsource;
}    
close(KEY) || die;
#&print_hash(%utt2subsource); exit;

#================================
# Handle cmn2 (.sph files)
#================================
my %dir = ();
my %tmp_dir = ();
my @source = ('cmn2','pstn','voip','vast');
foreach my $s (@source) {
    $dir{$s} = "$out_dir/sre18_dev_${s}";
    if (system("mkdir -p $dir{$s}")) {
	die "Error making directory $dir{$s}";
    }
    $tmp_dir{$s} = "$dir{$s}/tmp";
    if (system("mkdir -p $tmp_dir{$s}") != 0) {
	die "Error making directory $tmp_dir{$s}";
    }
}

if (system("find $db_base/data/ -name '*.sph' > $tmp_dir{'cmn2'}/sph.list") != 0) {
    die "Error getting list of sph files";
}

open(SPKR_CMN2, ">$dir{'cmn2'}/utt2spk") || die "Could not open the output file $dir{'cmn2'}/utt2spk";
open(WAV_CMN2, ">$dir{'cmn2'}/wav.scp") || die "Could not open the output file $dir{'cmn2'}/wav.scp";
open(SPKR_PSTN, ">$dir{'pstn'}/utt2spk") || die "Could not open the output file $dir{'pstn'}/utt2spk";
open(WAV_PSTN, ">$dir{'pstn'}/wav.scp") || die "Could not open the output file $dir{'pstn'}/wav.scp";
open(SPKR_VOIP, ">$dir{'voip'}/utt2spk") || die "Could not open the output file $dir{'voip'}/utt2spk";
open(WAV_VOIP, ">$dir{'voip'}/wav.scp") || die "Could not open the output file $dir{'voip'}/wav.scp";

open(SPHLIST, "<$tmp_dir{'cmn2'}/sph.list") or die "cannot open wav list";
while(<SPHLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  $utt=$t[$#t];
  print WAV_CMN2 "$utt"," sph2pipe -f wav -p -c 1 $sph |\n";
  print SPKR_CMN2 "$utt $utt\n";
  if (exists($utt2subsource{$utt})) {
      if ($utt2subsource{$utt} eq "pstn") {
	  print WAV_PSTN "$utt"," sph2pipe -f wav -p -c 1 $sph |\n";
	  print SPKR_PSTN "$utt $utt\n";
      } else {
	  if ($utt2subsource{$utt} eq "voip") {
	      print WAV_VOIP "$utt"," sph2pipe -f wav -p -c 1 $sph |\n";
	      print SPKR_VOIP "$utt $utt\n";
	  }
      }
  }
}
close(WAV_CMN2) || die;
close(SPKR_CMN2) || die;
close(WAV_PSTN) || die;
close(SPKR_PSTN) || die;
close(WAV_VOIP) || die;
close(SPKR_VOIP) || die;
close(SPHLIST) || die;


#================================
# Handle vast (.flac files)
#================================
if (system("find $db_base/data/ -name '*.flac' > $tmp_dir{'vast'}/flac.list") != 0) {
  die "Error getting list of flac files";
}

open(FLACLIST, "<$tmp_dir{'vast'}/flac.list") or die "cannot open flac list";
open(SPKR_VAST, ">$dir{'vast'}/utt2spk") || die "Could not open the output file $dir{'vast'}/utt2spk";
open(WAV_VAST, ">$dir{'vast'}/wav.scp") || die "Could not open the output file $dir{'vast'}/wav.scp";

while(<FLACLIST>) {
  chomp;
  $flac = $_;
  @t = split("/",$flac);
  $utt=$t[$#t];
  print WAV_VAST "$utt"," sox $flac -r 8000 -t wav - |\n";
  print SPKR_VAST "$utt $utt\n";
}
close(WAV_VAST) || die;
close(SPKR_VAST) || die;
close(FLACLIST) || die;



#================================
# Check files
#================================
foreach my $s (@source) {
    if (system("utils/utt2spk_to_spk2utt.pl $dir{$s}/utt2spk >$dir{$s}/spk2utt") != 0) {
	die "Error creating spk2utt file in directory $dir{$s}";
    }
    if (system("utils/fix_data_dir.sh $dir{$s}") != 0) {
	die "Error fixing data dir $dir{$s}";
    }
}


# Private function
sub print_hash {
    my (%hashtbl) = @_;
    foreach my $k (sort keys %hashtbl) {
        print "$k: $hashtbl{$k}\n";
    }
}
