This repository contains Kaldi and python scripts for two datasets:

1) Callhome (see v2-1/run.sh and v2-2/run.sh)
2) CUHK-NCD (see v2-1/run_cuhkncd.sh and v2-2/run_cuhkncd.sh)

## Performance (DER)

### Callhome:

No. of speakers unknown: 7.32%      
No. of speakers known:   6.29%             

### CUHK-NCD:

| VAD   |  Part |  Miss | FA    | Spk-err |  DER |
| :---: | :---: | :---: | :---: | :---: | :---:| 
| Aspire | 0  |    10.8 | 1.0 | 3.7  |     15.50 |
| Aspire | 1  |    10.6 | 0.9 | 4.6  |     16.04 |
